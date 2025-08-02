#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证原始 toolkit 与重构版 wakenLLM-toolkit 在相同数据 / 模型配置下的行为一致性。
"""

import subprocess
import sys
import os
import pathlib
import shutil
import re
import json
from ruamel.yaml import YAML

# =================================================================
# 1. 配置 (CONFIGURATION)
# =================================================================
PARENT_DIR = pathlib.Path(__file__).resolve().parent.parent
ORIGINAL_TOOLKIT_PATH = PARENT_DIR / "toolkit"
NEW_TOOLKIT_PATH = PARENT_DIR / "wakenLLM-toolkit"

# --- 测试用例配置 ---
MODEL_NAME = "qwen2.5-7b-instruct"
ORIGINAL_DATASET_NAME = "FLD"  # 使用哪个原始数据集
TEST_DATASET_NAME = "FLD_test"  # 临时测试文件的名字
SAMPLE_LIMIT = None # <--- 在这里控制测试样本的数量！设置为None则使用全部数据


class bcolors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


# =================================================================
# 2. 辅助函数 (HELPER FUNCTIONS)
# =================================================================

def print_header(title: str) -> None:
    print(f"\n{bcolors.HEADER}{'=' * 26} {title.upper()} {'=' * 26}{bcolors.ENDC}")


def prepare_test_data() -> None:
    """从原始数据集中切片，为两个工具包生成一个小的临时测试文件。"""
    limit_text = f"(limit: {SAMPLE_LIMIT} samples)" if SAMPLE_LIMIT is not None else "(full dataset)"
    print_header(f"Preparing Test Data {limit_text}")

    original_data_path = ORIGINAL_TOOLKIT_PATH / "task1-2" / f"{ORIGINAL_DATASET_NAME}.json"
    if not original_data_path.exists():
        raise FileNotFoundError(f"原始数据集未找到: {original_data_path}")

    with open(original_data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    test_data = full_data[:SAMPLE_LIMIT] if SAMPLE_LIMIT is not None else full_data

    # 将切片后的数据写入两个项目目录中
    test_data_path_old = ORIGINAL_TOOLKIT_PATH / "task1-2" / f"{TEST_DATASET_NAME}.json"
    test_data_path_new = NEW_TOOLKIT_PATH / "data" / f"{TEST_DATASET_NAME}.json"

    for path in [test_data_path_old, test_data_path_new]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)

    print(f"{bcolors.OKGREEN}Test data '{TEST_DATASET_NAME}.json' created successfully in both toolkits.{bcolors.ENDC}")


def run_command_stream(cmd, cwd: pathlib.Path, log_path: pathlib.Path) -> str:
    """边跑边看：把子进程的输出同时写到屏幕和 log 文件，返回完整日志文本。"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"{bcolors.OKCYAN}> Executing in '{cwd}': {' '.join(cmd)}{bcolors.ENDC}")

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
        finally:
            proc.wait()

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-40:]
        print(f"{bcolors.FAIL}❌ Sub-process exited with code {proc.returncode}.{bcolors.ENDC}")
        print("--- Last 40 log lines ---")
        print("\n".join(tail))
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    return log_path.read_text(encoding="utf-8")


# ... (clean_directories, extract_metrics, compare_metrics 等函数不变) ...
def clean_directories() -> None:
    print_header("Cleaning directories")
    for path in [
        ORIGINAL_TOOLKIT_PATH / "results",
        ORIGINAL_TOOLKIT_PATH / "task1-2" / "results",
        *NEW_TOOLKIT_PATH.glob("**/results"),
    ]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    print(f"{bcolors.OKGREEN}Directories cleaned.{bcolors.ENDC}")


METRIC_PATTERNS = {
    "tcr1": re.compile(r"--- Stage 1 Stimulation 评估结果 \(TCR¹\) ---.*?准确率:\s*([\d.]+)%", re.DOTALL),
    "tcr2": re.compile(r"--- Stage 2 Reflection 评估结果 \(TCR²\) ---.*?准确率:\s*([\d.]+)%",  re.DOTALL),
    "ocr":  re.compile(r"\*\*总体转换率 \(OCR\):\s*([\d.]+)%\*\*"),
}


def extract_metrics(log_text: str) -> dict[str, float]:
    metrics = {}
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(log_text)
        if m:
            metrics[k] = float(m.group(1))
    return metrics


def compare_metrics(old: dict[str, float], new: dict[str, float]) -> bool:
    ok = True
    if "tcr1" not in old or "tcr1" not in new:
        print(f"{bcolors.FAIL}❌ TCR¹ 未同时出现在两份日志中！{bcolors.ENDC}")
        return False
    if abs(old["tcr1"] - new["tcr1"]) > 0.01:
        print(f"{bcolors.FAIL}❌ TCR¹ 不一致：old={old['tcr1']}%, new={new['tcr1']}%{bcolors.ENDC}")
        ok = False
    else:
        print(f"{bcolors.OKGREEN}✅ TCR¹ 匹配：{old['tcr1']}%{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}ℹ️  New-toolkit 额外指标：TCR²={new.get('tcr2', 'N/A')}%, "
          f"OCR={new.get('ocr', 'N/A')}%{bcolors.ENDC}")
    return ok


# =================================================================
# 3. 运行两个工具链
# =================================================================

def run_old_system() -> str:
    print_header("Running ORIGINAL toolkit")
    log = ORIGINAL_TOOLKIT_PATH / "task1-2" / "run.log"
    cmd = ["bash", "run_settings1.sh",
           "--dataset", TEST_DATASET_NAME,  # 使用测试数据集
           "--model", MODEL_NAME,
           "--max_workers", "5"]
    return run_command_stream(cmd, ORIGINAL_TOOLKIT_PATH / "task1-2", log)


def run_new_system() -> str:
    print_header("Running REFACTORED toolkit")
    cfg_path = NEW_TOOLKIT_PATH / "configs" / "experiment.yaml"
    yaml = YAML()
    raw_yaml = cfg_path.read_text(encoding="utf-8")

    try:
        data = yaml.load(raw_yaml)
        data["model_name"] = MODEL_NAME
        data["dataset_name"] = TEST_DATASET_NAME  # 使用测试数据集
        data["run_tasks"] = ["vanilla"]
        yaml.dump(data, cfg_path.open("w", encoding="utf-8"))

        log = NEW_TOOLKIT_PATH / "run.log"
        cmd = ["python", "main.py", "--config", str(cfg_path)]
        return run_command_stream(cmd, NEW_TOOLKIT_PATH, log)
    finally:
        cfg_path.write_text(raw_yaml, encoding="utf-8")
        print(f"{bcolors.OKCYAN}Restored original experiment.yaml{bcolors.ENDC}")


# =================================================================
# 4. 主入口
# =================================================================

def main() -> None:
    try:
        clean_directories()
        prepare_test_data()  # 在运行前准备好测试数据
        old_log = run_old_system()
        new_log = run_new_system()

        print_header("Verification Phase")
        passed = compare_metrics(extract_metrics(old_log), extract_metrics(new_log))

        print_header("Final Verification Report")
        if passed:
            print(f"{bcolors.OKGREEN}{bcolors.BOLD}✅ VERIFICATION PASSED — 两套工具链行为一致！{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}{bcolors.BOLD}❌ VERIFICATION FAILED — 请检查上方差异！{bcolors.ENDC}")
    except Exception as e:
        print(f"{bcolors.FAIL}{bcolors.BOLD}发生异常：{e}{bcolors.ENDC}")


if __name__ == "__main__":
    main()