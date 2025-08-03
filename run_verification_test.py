#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import pathlib
import shutil
import re
from ruamel.yaml import YAML
import json
import pandas as pd  # 引入pandas来创建和显示表格

# =================================================================
# 1. 配置 (CONFIGURATION)
# =================================================================
# --- 路径配置 ---
PARENT_DIR = pathlib.Path(__file__).resolve().parent
ORIGINAL_TOOLKIT_PATH = PARENT_DIR.parent / "toolkit"
NEW_TOOLKIT_PATH = PARENT_DIR

# --- 实验矩阵配置 ---
# 在这里配置您想运行的所有数据集和任务
DATASETS_TO_RUN = ["FLD"]  # 您可以加入 "ScienceQA_language_arts", "ScienceQA_phy_bio"
TASKS_TO_RUN = ["vanilla", "rtg_process", "rtg_label"]

# --- 模型与样本配置 ---
MODEL_NAME = "qwen2.5-7b-instruct"
SAMPLE_LIMIT = 50  # 每个数据集取前n个样本进行测试

# --- Python解释器路径 (强制统一环境的关键) ---
# 请确保这里的路径是您新toolkit虚拟环境中的python解释器
PYTHON_EXE = "/Users/nianzhen/Desktop/wakenLLM-toolkit/.venv/bin/python3.12"  # <--- 请根据您的实际情况修改


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


def get_task_base_dir(dataset_name: str) -> pathlib.Path:
    """根据数据集名称确定其所在的task目录"""
    if "ScienceQA" in dataset_name:
        return ORIGINAL_TOOLKIT_PATH / "task3-4"
    else:
        return ORIGINAL_TOOLKIT_PATH / "task1-2"


def prepare_test_data(dataset_name: str, base_dir: pathlib.Path) -> None:
    """为指定的数据集准备测试数据文件"""
    test_dataset_name = f"{dataset_name}_test"
    print_header(f"Preparing Test Data for '{dataset_name}' (limit: {SAMPLE_LIMIT} samples)")
    original_data_path = base_dir / f"{dataset_name}.json"
    if not original_data_path.exists():
        print(f"{bcolors.WARNING}⚠️  原始数据集未找到: {original_data_path}，跳过...{bcolors.ENDC}")
        raise FileNotFoundError
    with open(original_data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    test_data = full_data[:SAMPLE_LIMIT]

    test_data_path_old = base_dir / f"{test_dataset_name}.json"
    test_data_path_new = NEW_TOOLKIT_PATH / "data" / f"{test_dataset_name}.json"

    test_data_path_new.parent.mkdir(exist_ok=True)  # 确保新toolkit的data目录存在

    with open(test_data_path_old, 'w', encoding='utf-8') as f: json.dump(test_data, f, indent=2)
    with open(test_data_path_new, 'w', encoding='utf-8') as f: json.dump(test_data, f, indent=2)
    print(f"{bcolors.OKGREEN}Test data '{test_dataset_name}.json' created successfully.{bcolors.ENDC}")


def run_command_stream(cmd, cwd: pathlib.Path, log_path: pathlib.Path, mode="w") -> str:
    """执行命令并流式传输输出，支持写入(w)或追加(a)模式"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print(f"{bcolors.OKCYAN}> Executing in '{cwd}': {' '.join(cmd)}{bcolors.ENDC}")
    with log_path.open(mode, encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
        finally:
            proc.wait()
    if proc.returncode != 0:
        print(f"{bcolors.FAIL}❌ Sub-process exited with code {proc.returncode}. Check log for details.{bcolors.ENDC}")
    return log_path.read_text(encoding="utf-8")


def clean_directories() -> None:
    """清理所有旧的result和log文件"""
    print_header("Cleaning all result directories")
    paths_to_clean = [
        ORIGINAL_TOOLKIT_PATH / "task1-2" / "results",
        ORIGINAL_TOOLKIT_PATH / "task3-4" / "results",
        NEW_TOOLKIT_PATH / "results",
    ]
    for path in paths_to_clean:
        if path.exists(): shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    # 清理旧的顶层日志
    if (NEW_TOOLKIT_PATH / "run.log").exists():
        (NEW_TOOLKIT_PATH / "run.log").unlink()

    print(f"{bcolors.OKGREEN}Directories cleaned.{bcolors.ENDC}")


def extract_metrics(log_text: str, toolkit_type: str, task_name: str) -> dict:
    """从日志文本中提取所有关键指标"""
    metrics = {"TCR¹": "N/A", "TCR²": "N/A", "OCR": "N/A", "RtG_Process": "N/A", "RtG_Label": "N/A"}

    if toolkit_type == 'new':
        if task_name == 'vanilla':
            patterns = {
                "TCR¹": re.compile(r"Stage 1 Stimulation.*?准确率:\s*([\d.]+)%", re.DOTALL),
                "TCR²": re.compile(r"Stage 2 Reflection.*?准确率:\s*([\d.]+)%", re.DOTALL),
                "OCR": re.compile(r"\*\*总体转换率 \(OCR\):\s*([\d.]+)%\*\*"),
            }
        elif task_name == 'rtg_process':
            match = re.search(r"--- RtG Process Conformity 最终评估结果 ---\s*.*?准确率:\s*([\d.]+)%", log_text,
                              re.DOTALL)
            if match: metrics["RtG_Process"] = f"{float(match.group(1)):.2f}%"
            patterns = {}
        elif task_name == 'rtg_label':
            # 提取所有情境的平均准确率
            accuracies = re.findall(r"--- Situation '.*?' 评估结果 ---\s*.*?准确率:\s*([\d.]+)%", log_text, re.DOTALL)
            if accuracies:
                avg_acc = sum(float(acc) for acc in accuracies) / len(accuracies)
                metrics["RtG_Label"] = f"{avg_acc:.2f}%"
            patterns = {}
    else:  # old toolkit
        if task_name == 'vanilla':
            patterns = {
                "TCR¹": re.compile(r"--- Stage 1 Stimulation 评估结果 \(TCR¹\) ---\s*.*?准确率: ([\d.]+)%", re.DOTALL),
                "TCR²": re.compile(r"Start Step 5\s*.*?准确率: ([\d.]+)%", re.DOTALL),
            }
        elif task_name == 'rtg_process':
            # 旧版的rtg_process最终准确率在step6的日志里
            match = re.search(r"Start Step 6[\s\S]*?准确率: ([\d.]+)%", log_text, re.DOTALL)
            if match: metrics["RtG_Process"] = f"{float(match.group(1)):.2f}%"
            patterns = {}
        elif task_name == 'rtg_label':
            # 旧版的rtg_label需要从多个step5的日志中提取并平均
            accuracies = re.findall(r"Start Step 5[\s\S]*?准确率: ([\d.]+)%", log_text, re.DOTALL)
            if accuracies:
                avg_acc = sum(float(acc) for acc in accuracies) / len(accuracies)
                metrics["RtG_Label"] = f"{avg_acc:.2f}%"
            patterns = {}

    for key, pattern in patterns.items():
        match = pattern.search(log_text)
        if match:
            metrics[key] = f"{float(match.group(1)):.2f}%"

    return metrics


def run_old_toolkit(dataset_name: str, task_name: str, base_dir: pathlib.Path) -> str:
    """根据任务名称，在统一环境下运行旧toolkit的相应流程"""
    print_header(f"Running OLD toolkit for: {dataset_name} / {task_name}")
    log_path = base_dir / "run.log"
    if log_path.exists(): log_path.unlink()  # 每次都创建新日志

    test_dataset_name = f"{dataset_name}_test"

    if task_name == "vanilla":
        scripts = ["step1.py", "step2.py", "step3.py", "step4_settings1.py", "step5_settings1.py"]
        for i, script in enumerate(scripts):
            print(f"\n--- [Vanilla] Running Step {i + 1}/{len(scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME]
            if "step3.py" not in script:
                cmd.extend(["--max_workers", "5"])
            run_command_stream(cmd, base_dir, log_path, mode="a")

    elif task_name == "rtg_process":
        # 预处理是共享的，需要先运行
        pre_scripts = ["step1.py", "step2.py", "step3.py"]
        for i, script in enumerate(pre_scripts):
            print(f"\n--- [RtG_Process Pre-req] Running Step {i + 1}/{len(pre_scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME]
            if "step3.py" not in script:
                cmd.extend(["--max_workers", "5"])
            run_command_stream(cmd, base_dir, log_path, mode="a")

        # 运行rtg_process的核心脚本
        scripts = ["step4_settings3.py", "step5_settings3.py", "step6_settings3.py"]
        for i, script in enumerate(scripts):
            print(f"\n--- [RtG_Process] Running Step {i + 1}/{len(scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME, "--max_workers", "5"]
            run_command_stream(cmd, base_dir, log_path, mode="a")

    elif task_name == "rtg_label":
        # 预处理是共享的，需要先运行
        pre_scripts = ["step1.py", "step2.py", "step3.py"]
        for i, script in enumerate(pre_scripts):
            print(f"\n--- [RtG_Label Pre-req] Running Step {i + 1}/{len(pre_scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME]
            if "step3.py" not in script:
                cmd.extend(["--max_workers", "5"])
            run_command_stream(cmd, base_dir, log_path, mode="a")

        # 运行rtg_label的核心脚本，遍历所有情境
        situations = ["All_Wrong", "2Over3_Wrong", "Half_Wrong", "All_Right"]
        for situation in situations:
            print(f"\n--- [RtG_Label] Processing Situation: {situation} ---")
            cmd4 = [PYTHON_EXE, "step4_settings2.py", "--dataset", test_dataset_name, "--model", MODEL_NAME,
                    "--situation", situation, "--max_workers", "5"]
            run_command_stream(cmd4, base_dir, log_path, mode="a")
            cmd5 = [PYTHON_EXE, "step5_settings2.py", "--dataset", test_dataset_name, "--model", MODEL_NAME,
                    "--situation", situation, "--max_workers", "5"]
            run_command_stream(cmd5, base_dir, log_path, mode="a")
    else:
        raise ValueError(f"Unknown task for old toolkit: {task_name}")

    return log_path.read_text(encoding="utf-8")


def run_new_toolkit(dataset_name: str, task_name: str) -> str:
    """根据任务名称，配置并运行新toolkit"""
    print_header(f"Running NEW toolkit for: {dataset_name} / {task_name}")
    cfg_path = NEW_TOOLKIT_PATH / "configs" / "experiment.yaml"
    log_path = NEW_TOOLKIT_PATH / "run.log"
    if log_path.exists(): log_path.unlink()

    yaml = YAML()
    original_yaml_text = cfg_path.read_text(encoding="utf-8")
    try:
        data = yaml.load(original_yaml_text)
        data["model_name"] = MODEL_NAME
        data["dataset_name"] = f"{dataset_name}_test"
        data["run_tasks"] = [task_name]
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f)

        cmd = [PYTHON_EXE, "main.py", "--config", str(cfg_path)]
        return run_command_stream(cmd, NEW_TOOLKIT_PATH, log_path)
    finally:
        cfg_path.write_text(original_yaml_text, encoding="utf-8")
        print(f"{bcolors.OKCYAN}Restored original experiment.yaml{bcolors.ENDC}")


# =================================================================
# 3. 主流程 (MAIN WORKFLOW)
# =================================================================
def main() -> None:
    # 清理工作区
    clean_directories()

    # 用于存储所有结果的列表
    results_summary = []

    # --- 主循环：遍历所有数据集和任务 ---
    for dataset in DATASETS_TO_RUN:
        task_base_dir = get_task_base_dir(dataset)
        try:
            prepare_test_data(dataset, task_base_dir)
        except FileNotFoundError:
            continue  # 如果原始数据文件找不到，就跳过这个数据集

        for task in TASKS_TO_RUN:
            # 运行旧版
            old_log = run_old_toolkit(dataset, task, task_base_dir)
            old_metrics = extract_metrics(old_log, 'old', task)
            old_metrics["Toolkit"] = "Old"
            old_metrics["Dataset"] = dataset
            old_metrics["Task"] = task
            results_summary.append(old_metrics)

            # 运行新版
            new_log = run_new_toolkit(dataset, task)
            new_metrics = extract_metrics(new_log, 'new', task)
            new_metrics["Toolkit"] = "New"
            new_metrics["Dataset"] = dataset
            new_metrics["Task"] = task
            results_summary.append(new_metrics)

    # --- 生成最终报告 ---
    print_header("Final Comprehensive Report")
    if not results_summary:
        print("No results were generated.")
        return

    # 使用pandas创建漂亮的表格
    df = pd.DataFrame(results_summary)

    # 重新排列和填充列，以获得最佳的可读性
    display_columns = ["Dataset", "Task", "Toolkit", "TCR¹", "TCR²", "OCR", "RtG_Process", "RtG_Label"]
    df_display = pd.DataFrame(columns=display_columns)
    for col in display_columns:
        if col in df.columns:
            df_display[col] = df[col]
        else:
            df_display[col] = "N/A"

    df_display.fillna("N/A", inplace=True)

    print(df_display.to_string())

    # --- 将报告保存到文件 ---
    report_path = PARENT_DIR / "final_comparison_report.csv"
    df_display.to_csv(report_path, index=False)
    print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}✅ 报告已保存到: {report_path}{bcolors.ENDC}")


if __name__ == "__main__":
    main()