# ==============================================================================
# Makefile for the WAKENLLM Toolkit (Professional Edition)
#
# This version incorporates advanced features for robustness, safety, and
# meticulous logging, suitable for rigorous academic and production use.
#
# @author: Junqi Yang
# @version: 2.0.0
# ==============================================================================

# --- Advanced Shell Configuration ---
# Use bash for all commands and enforce strict error checking.
SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# --- Configuration ---
PYTHON := $(shell command -v python3)
MODEL ?= "gpt-3.5-turbo-1106"
DATASET ?= "FLD"

# --- Dynamic File & Directory Paths ---
CONFIG_FILE := configs/experiment.yaml
RESULTS_DIR := results
DATA_DIR := data
LOGS_DIR := logs
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# --- Self-Documentation ---
.PHONY: help
help:
	@echo "===================== WAKENLLM Toolkit Controller ====================="
	@echo "Usage: make [target] [VARIABLE=value]"
	@echo ""
	@echo "Available Targets:"
	@echo "  setup              Install all project dependencies."
	@echo "  download-data      Download required datasets (Note: FLD must be handled manually)."
	@echo "  run-vanilla        Run the Vanilla Pipeline experiment on the $(DATASET) dataset."
	@echo "  run-rtg-label      Run the RtG Label Conformity experiment on the $(DATASET) dataset."
	@echo "  run-rtg-process    Run the RtG Process Conformity experiment on the $(DATASET) dataset."
	@echo "  run-all            Run all three experiments sequentially on the $(DATASET) dataset."
	@echo "  run-subset         Run the given task in one subset (make run-subset DATASET=FLD SAMPLES=50 TASK=vanilla)"
	@echo "  clean              Remove all generated files (results, logs, caches)."
	@echo ""
	@echo "Options:"
	@echo "  MODEL=<model_name>   Specify the model to use (default: $(MODEL))."
	@echo "  DATASET=<name>       Specify the dataset to use (default: $(DATASET))."
	@echo "====================================================================="

# --- Project Lifecycle Management ---
.PHONY: setup
setup:
	@echo "--> Setting up project environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "--> Environment setup complete."

.PHONY: download-data
download-data:
	@echo "--> Preparing datasets in [$(DATA_DIR)/]..."
	@mkdir -p $(DATA_DIR)
	@echo "NOTE: Per its license, the FLD dataset must be downloaded manually from its source."
	@if [ ! -f "$(DATA_DIR)/FOLIO.json" ]; then \
		echo "Downloading FOLIO dataset..."; \
		wget -q --show-progress -O $(DATA_DIR)/FOLIO.json https://raw.githubusercontent.com/Yifan-Song793/FOLIO/main/data/folio.json; \
	else \
		echo "FOLIO dataset already exists. Skipping download."; \
	fi
	@echo "--> Dataset preparation finished."

.PHONY: run-subset
run-subset:
	@# 检查 SAMPLES 变量是否被定义
	@if [ -z "$(SAMPLES)" ]; then \
		echo "错误: 请指定样本数量. 用法: make run-subset DATASET=<name> SAMPLES=<number> TASK=<task_name>"; \
		exit 1; \
	fi
	@# 检查 TASK 变量是否被定义
	@if [ -z "$(TASK)" ]; then \
		echo "错误: 请指定要运行的任务. 用法: make run-subset ... TASK=<vanilla|rtg_label|rtg_process>"; \
		exit 1; \
	fi
	@echo "--> 第一步: 准备一个包含 $(SAMPLES) 条样本的 '$(DATASET)' 数据子集..."
	$(PYTHON) scripts/prepare_subset.py --dataset $(DATASET) --samples $(SAMPLES)

	@echo "\n--> 第二步: 运行实验..."
	$(eval SUBSET_NAME := $(DATASET)_subset_$(SAMPLES))
	$(eval ORIGINAL_DATASET := $(DATASET)) # 保存原始的DATASET值
	@# 临时将DATASET变量设置为子集的名称，并运行实验
	$(MAKE) run-vanilla DATASET=$(SUBSET_NAME) MODEL=$(MODEL)
	@# 这里我们调用已有的 run-vanilla, run-rtg-label 等目标，但传入了新的DATASET值
# --- Experiment Execution Engine (Upgraded) ---
# This version uses a secure temporary file for configs and timestamped logs.
# Arguments: 1=task_name
define run_experiment
	@mkdir -p $(LOGS_DIR)
	$(eval TMP_CFG := $(shell mktemp /tmp/wkn_llm_config.XXXXXX.yaml))
	$(eval LOG_FILE := $(LOGS_DIR)/$(TIMESTAMP)_$(1)_$(DATASET)_$(MODEL).log)
	@echo "--> Starting experiment: $(1) | Log file: $(LOG_FILE)"
	@# Create a temporary config file safely
	@sed \
	    -e 's/^run_tasks:.*/run_tasks: ["$(1)"]/' \
	    -e 's/^dataset_name:.*/dataset_name: "$(DATASET)"/' \
	    -e 's/^model_name:.*/model_name: "$(MODEL)"/' \
	    $(CONFIG_FILE) > $(TMP_CFG)
	@# Execute and pipe output to both console and the unique log file
	$(PYTHON) main.py --config $(TMP_CFG) | tee $(LOG_FILE)
	@# Clean up the temporary file
	@rm -f $(TMP_CFG)
	@echo "--> Experiment '$(1)' finished."
endef

.PHONY: run-vanilla
run-vanilla:
	$(call run_experiment,vanilla)

.PHONY: run-rtg-label
run-rtg-label:
	$(call run_experiment,rtg_label)

.PHONY: run-rtg-process
run-rtg-process:
	$(call run_experiment,rtg_process)

.PHONY: run-all
run-all: run-vanilla run-rtg-label run-rtg-process

# --- Cleanup ---
.PHONY: clean
clean:
	@echo "--> Cleaning up generated files..."
	@rm -rf $(RESULTS_DIR)
	@rm -rf $(LOGS_DIR)
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "--> Cleanup complete."

.DEFAULT_GOAL := help