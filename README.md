# WAKENLLM Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the AAAI paper: **"WAKENLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking"**.

This toolkit provides a modular and extensible framework to reproduce all experiments in the paper, including the identification of the "Vague Perception" phenomenon and the execution of the multi-stage stimulation and reflection pipelines.

## 🌟 Features

* **Modular Architecture**: A clean, object-oriented design that separates concerns (data handling, model interaction, evaluation, etc.).
* **Configuration Driven**: Easily manage and run complex experiments by simply editing YAML configuration files.
* **Extensible**: Designed to be easily extended with new datasets, models, or experimental pipelines.
* **Reproducibility**: Implements the full end-to-end workflows for:
    * The **Vanilla Pipeline** (Stage 1 Stimulation & Stage 2 Reflection).
    * **Remind-then-Guide (RtG) Label Conformity** tests.
    * **Remind-then-Guide (RtG) Process Conformity** tests.

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/WAKENLLM-Toolkit.git](https://github.com/YourUsername/WAKENLLM-Toolkit.git)
    cd WAKENLLM-Toolkit
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt.txt
    ```

## ⚙️ Configuration

Before running any experiments, you need to configure your API keys and experiment parameters.

### 1. API Keys

This project requires API access to one or more LLMs.

1.  Copy the template file `secrets.template.yaml` to a new file named `secrets.yaml`:
    ```bash
    cp secrets.template.yaml secrets.yaml
    ```
2.  Open `secrets.yaml` and fill in your own API key and base URL. This file is ignored by Git and will not be uploaded.

### 2. Experiment Parameters

All experiment settings are controlled via the `configs/experiment.yaml` file. You can modify this file or create new ones.

Key parameters include:
* `model_name`: The name of the model you want to test.
* `dataset_name`: The dataset to use (`FLD`, `FOLIO`, `ScienceQA_phy_bio`, etc.).
* `run_tasks`: A list of experiments to run. Options are: `"vanilla"`, `"rtg_label"`, `"rtg_process"`.

## 🚀 Quick Start

1.  **Prepare Data**: Download the necessary datasets (see Data Preparation section) and place them in the `data/` directory.
2.  **Configure**: Ensure your `secrets.yaml` and `configs/experiment.yaml` files are correctly set up.
3.  **Run**: Execute the main script from the project root directory:
    ```bash
    python main.py --config configs/experiment.yaml
    ```
    The toolkit will start running the tasks specified in your config file, and all results will be saved to the `results/` directory.

## 📚 Data Preparation

The toolkit expects the raw dataset files to be placed in the `data/` directory. You can download the datasets from their original sources:

* **FLD**: [Link to FLD dataset source]
* **FOLIO**: [Link to FOLIO dataset source]
* **ScienceQA**: [Link to ScienceQA dataset source]

Please ensure the downloaded files are named to match the `dataset_name` used in the configuration (e.g., `data/FLD.json`).

## 🏛️ Project Structure

The toolkit follows a clean, modular structure:
```
.
├── configs/            # Experiment configuration files
├── data/               # Raw dataset files (user-provided)
├── results/            # All generated outputs and evaluation results
├── src/                # All source code
│   ├── config_loader.py  # Loads configuration
│   ├── data_handler.py   # Handles all file I/O
│   ├── evaluator.py      # Calculates all metrics
│   ├── llm_handler.py    # Manages all LLM API interactions
│   └── pipeline.py       # The core experimental workflow
├── main.py             # Main entry point of the toolkit
├── requirements.txt    # Project dependencies
├── README.md           # This file
└── LICENSE             # MIT License
```

## 📜 Citation

If you use this toolkit or the WAKENLLM framework in your research, please cite our paper:

```bibtex
@inproceedings{your-lastname_wakenllm_2025,
    title = {WAKENLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking},
    author = {Your Name and Coauthor Name(s)},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    year = {2025}
}
```
*(Please update the BibTeX entry with the correct author names and official publication details when available.)*

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.