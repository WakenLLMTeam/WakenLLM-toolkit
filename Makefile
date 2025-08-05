# ==============================================================================
# Makefile for the WAKENLLM Toolkit
#
# This Makefile provides a unified and reproducible command-line interface for
# managing the project's lifecycle, from setup to execution and cleanup.
# It is designed to be portable and independent of the user's shell.
#
# @author: Gemini AI
# @version: 1.0.0
# ==============================================================================

# --- Configuration ---
# Use the python interpreter from the current environment. This ensures that
# if the user is in a virtual environment (e.g., venv), we use it.
PYTHON := $(shell command -v python3)

# Default model name. Can be overridden from the command line.
# Example: make run-vanilla MODEL="gpt-4o"
MODEL ?= "gpt-3.5-turbo-1106"
DATASET ?= "FLD"

# Directories
CONFIG_FILE := configs/experiment.yaml
RESULTS_DIR := results
DATA_DIR := data
LOG_FILE := run.log

# --- Self-Documentation ---
# The 'help' target is the default. It prints a list of available commands.
# It's a common best practice for discoverability.
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
	@echo "  visualize          (Future) Generate and display visualizations from results."
	@echo "  clean              Remove all generated files (results, logs, caches)."
	@echo ""
	@echo "Options:"
	@echo "  MODEL=<model_name>   Specify the model to use (default: $(MODEL))."
	@echo "  DATASET=<name>       Specify the dataset to use (default: $(DATASET))."
	@echo "====================================================================="

# --- Project Lifecycle Management ---

# Target to set up the entire environment.
.PHONY: setup
setup:
	@echo "--> Setting up project environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "--> Environment setup complete."

# Target for downloading datasets.
# This is idempotent; it checks if files exist before downloading.
.PHONY: download-data
download-data:
	@echo "--> Preparing datasets in [$(DATA_DIR)/]..."
	@mkdir -p $(DATA_DIR)
	@echo "NOTE: Per its license, the FLD dataset must be downloaded manually from its source."
	# Example for FOLIO dataset
	@if [ ! -f "$(DATA_DIR)/FOLIO.json" ]; then \
		echo "Downloading FOLIO dataset..."; \
		wget -q --show-progress -O $(DATA_DIR)/FOLIO.json https://raw.githubusercontent.com/Yifan-Song793/FOLIO/main/data/folio.json; \
	else \
		echo "FOLIO dataset already exists. Skipping download."; \
	fi
	@echo "--> Dataset preparation finished."

# --- Experiment Execution ---

# A private helper function to run experiments. This avoids code duplication.
# It temporarily modifies the config file, runs the main script, and then restores the original config.
# Arguments: 1=task_name
define run_experiment
	@echo "--> Starting experiment: TASK=$(1), DATASET=$(DATASET), MODEL=$(MODEL)"
	@cp $(CONFIG_FILE) $(CONFIG_FILE).bak
	@# Use sed to replace the relevant lines in the config file. This is more robust than simple text replacement.
	@sed -e 's/^run_tasks:.*/run_tasks: ["$(1)"]/' \
	    -e 's/^dataset_name:.*/dataset_name: "$(DATASET)"/' \
	    -e 's/^model_name:.*/model_name: "$(MODEL)"/' \
	    $(CONFIG_FILE).bak > $(CONFIG_FILE)
	@# Execute the main script
	$(PYTHON) main.py --config $(CONFIG_FILE) | tee -a $(LOG_FILE)
	@# Restore the original config file
	@mv $(CONFIG_FILE).bak $(CONFIG_FILE)
	@echo "--> Experiment '$(1)' finished. Check [$(RESULTS_DIR)/] for output."
endef

# Public targets for running each type of experiment.
.PHONY: run-vanilla
run-vanilla:
	$(call run_experiment,vanilla)

.PHONY: run-rtg-label
run-rtg-label:
	$(call run_experiment,rtg_label)

.PHONY: run-rtg-process
run-rtg-process:
	$(call run_experiment,rtg_process)

# A convenience target to run all experiments for a given dataset.
.PHONY: run-all
run-all: run-vanilla run-rtg-label run-rtg-process

# --- Visualization and Cleanup ---

# Placeholder for the results visualization script.
.PHONY: visualize
visualize:
	@echo "--> Generating visualizations from results..."
	$(PYTHON) scripts/visualize.py

# Clean up all generated artifacts to ensure a fresh start.
.PHONY: clean
clean:
	@echo "--> Cleaning up generated files..."
	@rm -rf $(RESULTS_DIR)
	@rm -f $(LOG_FILE)
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "--> Cleanup complete."

# Set the default goal to 'help' so that running 'make' without arguments prints the help message.
.DEFAULT_GOAL := help