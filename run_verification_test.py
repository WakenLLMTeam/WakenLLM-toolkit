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
import pandas as pd  # Import pandas to create and display tables

# =================================================================
# 1. Configuration (CONFIGURATION)
# =================================================================
# --- Path configuration ---
PARENT_DIR = pathlib.Path(__file__).resolve().parent
ORIGINAL_TOOLKIT_PATH = PARENT_DIR.parent / "toolkit"
NEW_TOOLKIT_PATH = PARENT_DIR

# --- Experiment matrix configuration ---
# Configure all datasets and tasks you want to run here
DATASETS_TO_RUN = ["FLD"]  # You can add "ScienceQA_language_arts", "ScienceQA_phy_bio"
TASKS_TO_RUN = ["vanilla", "rtg_process", "rtg_label"]

# --- Model and sample configuration ---
MODEL_NAME = "qwen2.5-7b-instruct"
SAMPLE_LIMIT = 50  # Take first n samples from each dataset for testing

# --- Python interpreter path (key to forcing unified environment) ---
# Please ensure this path is the python interpreter in your new toolkit virtual environment
PYTHON_EXE = "/Users/nianzhen/Desktop/wakenLLM-toolkit/.venv/bin/python3.12"  # <--- Please modify according to your actual situation


class bcolors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


# =================================================================
# 2. Helper Functions (HELPER FUNCTIONS)
# =================================================================
def print_header(title: str) -> None:
    print(f"\n{bcolors.HEADER}{'=' * 26} {title.upper()} {'=' * 26}{bcolors.ENDC}")


def get_task_base_dir(dataset_name: str) -> pathlib.Path:
    """Determine the task directory based on dataset name"""
    if "ScienceQA" in dataset_name:
        return ORIGINAL_TOOLKIT_PATH / "task3-4"
    else:
        return ORIGINAL_TOOLKIT_PATH / "task1-2"


def prepare_test_data(dataset_name: str, base_dir: pathlib.Path) -> None:
    """Prepare test data files for the specified dataset"""
    test_dataset_name = f"{dataset_name}_test"
    print_header(f"Preparing Test Data for '{dataset_name}' (limit: {SAMPLE_LIMIT} samples)")
    original_data_path = base_dir / f"{dataset_name}.json"
    if not original_data_path.exists():
        print(f"{bcolors.WARNING}⚠️  Original dataset not found: {original_data_path}, skipping...{bcolors.ENDC}")
        raise FileNotFoundError
    with open(original_data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    test_data = full_data[:SAMPLE_LIMIT]

    test_data_path_old = base_dir / f"{test_dataset_name}.json"
    test_data_path_new = NEW_TOOLKIT_PATH / "data" / f"{test_dataset_name}.json"

    test_data_path_new.parent.mkdir(exist_ok=True)  # Ensure new toolkit's data directory exists

    with open(test_data_path_old, 'w', encoding='utf-8') as f: json.dump(test_data, f, indent=2)
    with open(test_data_path_new, 'w', encoding='utf-8') as f: json.dump(test_data, f, indent=2)
    print(f"{bcolors.OKGREEN}Test data '{test_dataset_name}.json' created successfully.{bcolors.ENDC}")


def run_command_stream(cmd, cwd: pathlib.Path, log_path: pathlib.Path, mode="w") -> str:
    """Execute command and stream output, supporting write(w) or append(a) mode"""
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

    # Clean old top-level logs
    if (NEW_TOOLKIT_PATH / "run.log").exists():
        (NEW_TOOLKIT_PATH / "run.log").unlink()

    print(f"{bcolors.OKGREEN}Directories cleaned.{bcolors.ENDC}")


def extract_metrics(log_text: str, toolkit_type: str, task_name: str) -> dict:
    """Extract all key metrics from log text"""
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
            # Extract average accuracy across all situations
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
            # Old version's rtg_process final accuracy is in step6 logs
            match = re.search(r"Start Step 6[\s\S]*?准确率: ([\d.]+)%", log_text, re.DOTALL)
            if match: metrics["RtG_Process"] = f"{float(match.group(1)):.2f}%"
            patterns = {}
        elif task_name == 'rtg_label':
            # Old version's rtg_label needs to extract and average from multiple step5 logs
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
    """Run the corresponding workflow of the old toolkit in a unified environment based on task name"""
    print_header(f"Running OLD toolkit for: {dataset_name} / {task_name}")
    log_path = base_dir / "run.log"
    if log_path.exists(): log_path.unlink()  # Create new log each time

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
        # Preprocessing is shared, need to run first
        pre_scripts = ["step1.py", "step2.py", "step3.py"]
        for i, script in enumerate(pre_scripts):
            print(f"\n--- [RtG_Process Pre-req] Running Step {i + 1}/{len(pre_scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME]
            if "step3.py" not in script:
                cmd.extend(["--max_workers", "5"])
            run_command_stream(cmd, base_dir, log_path, mode="a")

        # Run rtg_process core scripts
        scripts = ["step4_settings3.py", "step5_settings3.py", "step6_settings3.py"]
        for i, script in enumerate(scripts):
            print(f"\n--- [RtG_Process] Running Step {i + 1}/{len(scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME, "--max_workers", "5"]
            run_command_stream(cmd, base_dir, log_path, mode="a")

    elif task_name == "rtg_label":
        # Preprocessing is shared, need to run first
        pre_scripts = ["step1.py", "step2.py", "step3.py"]
        for i, script in enumerate(pre_scripts):
            print(f"\n--- [RtG_Label Pre-req] Running Step {i + 1}/{len(pre_scripts)}: {script} ---")
            cmd = [PYTHON_EXE, script, "--dataset", test_dataset_name, "--model", MODEL_NAME]
            if "step3.py" not in script:
                cmd.extend(["--max_workers", "5"])
            run_command_stream(cmd, base_dir, log_path, mode="a")

        # Run rtg_label core scripts, iterate through all situations
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
    """Configure and run new toolkit based on task name"""
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
# 3. Main Workflow (MAIN WORKFLOW)
# =================================================================
def main() -> None:
    # Clean workspace
    clean_directories()

    # List to store all results
    results_summary = []

    # --- Main loop: iterate through all datasets and tasks ---
    for dataset in DATASETS_TO_RUN:
        task_base_dir = get_task_base_dir(dataset)
        try:
            prepare_test_data(dataset, task_base_dir)
        except FileNotFoundError:
            continue  # Skip this dataset if original data file not found

        for task in TASKS_TO_RUN:
            # Run old version
            old_log = run_old_toolkit(dataset, task, task_base_dir)
            old_metrics = extract_metrics(old_log, 'old', task)
            old_metrics["Toolkit"] = "Old"
            old_metrics["Dataset"] = dataset
            old_metrics["Task"] = task
            results_summary.append(old_metrics)

            # Run new version
            new_log = run_new_toolkit(dataset, task)
            new_metrics = extract_metrics(new_log, 'new', task)
            new_metrics["Toolkit"] = "New"
            new_metrics["Dataset"] = dataset
            new_metrics["Task"] = task
            results_summary.append(new_metrics)

    # --- Generate final report ---
    print_header("Final Comprehensive Report")
    if not results_summary:
        print("No results were generated.")
        return

    # Use pandas to create beautiful tables
    df = pd.DataFrame(results_summary)

    # Rearrange and fill columns for optimal readability
    display_columns = ["Dataset", "Task", "Toolkit", "TCR¹", "TCR²", "OCR", "RtG_Process", "RtG_Label"]
    df_display = pd.DataFrame(columns=display_columns)
    for col in display_columns:
        if col in df.columns:
            df_display[col] = df[col]
        else:
            df_display[col] = "N/A"

    df_display.fillna("N/A", inplace=True)

    print(df_display.to_string())

    # --- Save report to file ---
    report_path = PARENT_DIR / "final_comparison_report.csv"
    df_display.to_csv(report_path, index=False)
    print(f"\n{bcolors.OKGREEN}{bcolors.BOLD}✅ Report saved to: {report_path}{bcolors.ENDC}")


if __name__ == "__main__":
    main()