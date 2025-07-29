# WAKENLLM Toolkit 设计文档

**版本**: 1.0
**日期**: 2025年7月28日

**以下所有内容由 Gemini-2.5pro生成，请注意审查**

## 1. 简介与核心设计哲学

WAKENLLM Toolkit是一个用于复现AAAI论文《WAKENLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking》中所有实验的官方工具包。

本工具包的设计遵循了现代软件工程中的核心原则——**关注点分离 (Separation of Concerns)**。其目的是将一个庞大、由多个脚本驱动的复杂实验流程，重构为一个模块化、配置驱动、易于理解、维护和扩展的健壮系统。每个模块都职责专一，协同工作，共同完成复杂的实验任务。

## 2. 系统架构与组件职责

工具包由以下七个核心组件构成，各司其职：

| 组件 | 职责 (做什么？) | 核心价值 |
| :--- | :--- | :--- |
| **`main.py`** | **启动器 (Launcher)**<br>项目的唯一入口。负责解析命令行，加载配置，然后“组装”并“启动”`Pipeline`。 | 提供一个统一、简单的用户交互界面。 |
| **`configs/`** | **控制中心 (Control Center)**<br>存放所有实验参数的`.yaml`文件。用户通过修改配置文件来“指挥”实验。 | 将**易变的参数**与**稳定的代码**分离，极大提升了实验的灵活性和可复现性。 |
| **`src/config_loader.py`** | **配置秘书 (Config Secretary)**<br>负责安全地读取`configs/`和`secrets.yaml`，并将它们合并成一个统一的配置对象。 | 实现了安全的密钥管理，从根本上杜绝了敏感信息泄露的风险。 |
| **`src/data_handler.py`** | **文件管家 (File Butler)**<br>负责所有文件的读写操作，包括数据集加载和结果保存。 | 将繁琐的文件I/O逻辑从核心算法中剥离，保证了文件管理的统一和整洁。 |
| **`src/llm_handler.py`** | **API外交官 (API Diplomat)**<br>封装了所有与大语言模型API的复杂网络交互，包括并发控制、错误处理和认证。 | 隔离了网络请求的复杂性，让`Pipeline`可以专注于“问什么”，而不是“怎么问”。 |
| **`src/evaluator.py`** | **裁判员 (The Judge)**<br>负责所有性能指标的计算，如准确率、OCR、CGR等。 | 将“如何评估”的逻辑从“如何执行实验”的逻辑中分离，使评估标准清晰独立。 |
| **`src/pipeline.py`** | **大脑与心脏 (The Brain & Heart)**<br>项目的核心，编排所有其他模块，实现了论文中定义的所有实验工作流。 | 这是您科学贡献最直接的代码体现，将复杂的实验步骤转化为清晰、自动化的代码流程。 |

## 3. 工作流程与数据流

一个典型的实验（如 Vanilla Pipeline）在系统中的数据流动路径如下：

1.  **启动**: 用户在命令行运行 `python main.py --config configs/my_exp.yaml`。
2.  **配置加载**: `main.py` 调用 `Config Loader` 读取配置文件和密钥，生成统一的 `config` 对象。
3.  **模块初始化**: `main.py` 使用 `config` 对象，初始化 `DataHandler`, `LLMHandler`, `Evaluator` 和 `Pipeline` 四大核心模块。
4.  **流程启动**: `main.py` 调用 `Pipeline` 的 `run()` 方法。
5.  **任务路由**: `Pipeline` 的 `run()` 方法检查 `config`，决定启动一个或多个实验工作流（如 `run_vanilla_experiment`）。
6.  **预处理**:
    * `Pipeline` 调用 `DataHandler` 加载原始数据集。
    * `Pipeline` 构建 `Prompts` 并交给 `LLMHandler` 进行第一轮推理，以识别出所有“模糊感知”样本。
7.  **核心处理**:
    * `Pipeline` 将识别出的“模糊感知”样本送入**Stage 1 Stimulation**。它再次构建新的刺激性`Prompts`，并交由`LLMHandler`处理。
    * `Pipeline` 将第一阶段失败的样本送入**Stage 2 Reflection**，重复上述过程。
8.  **评估与保存**: 在每个阶段，`Pipeline` 都会调用 `Evaluator` 来计算性能指标（如TCR¹, TCR²），并调用 `DataHandler` 将处理过的样本和评估结果保存到 `results/` 目录中。
9.  **最终总结**: 所有核心流程结束后，`Pipeline` 进行最终的汇总计算（如OCR），并完成整个实验。

## 4. `pipeline.py` 内部结构

作为项目的核心，`pipeline.py` 内部也遵循清晰的结构划分，主要分为四个功能区：

* **第一部分：决策中心 (`run` 方法)**
    * 作为唯一的公共入口，负责解析配置，并像路由器一样，将任务分发给不同的独立工作流。

* **第二部分：独立工作流 (`run_..._experiment` 方法)**
    * 包含三个并列的、独立的“生产线”方法：`run_vanilla_experiment`, `run_rtg_label_experiment`, `run_rtg_process_experiment`。每个方法都完整地封装了一个端到端的实验流程。

* **第三部分：生产工位 (`_`开头的私有方法)**
    * 构成每条生产线的具体操作步骤，如 `_get_all_vague_samples`, `_run_stage1_stimulation` 等。这些方法是可复用的组件，被不同的工作流按需调用。

* **第四部分：工具房 (Prompt工厂)**
    * 一个统一的 `_build_prompt` 方法和多个具体的 `_build_..._prompt` 方法。它将所有精心设计的Prompt模板集中管理，便于查阅、修改和扩展。
