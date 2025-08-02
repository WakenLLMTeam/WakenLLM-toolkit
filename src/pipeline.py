import asyncio
from typing import List, Dict, Any, Tuple

# 导入项目的所有依赖模块
from .data_handler import DataHandler
from .llm_handler import LLMHandler
from .evaluator import Evaluator


# --- 修正：确保类名与 main.py 中导入的名称一致 ---
class WakenllmPipeline:
    """
    WAKENLLM实验框架的核心流程。
    负责编排数据处理、模型调用和评估等所有步骤。
    """

    def __init__(self, config: Dict[str, Any], data_handler: DataHandler, llm_handler: LLMHandler,
                 evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator
        print("WakenllmPipeline 初始化完成。")

    # =================================================================
    # 1. 总运行入口 (RUN METHOD) - 决策中心
    # =================================================================
    async def run(self):
        """
        根据配置，路由到不同的实验流程。
        """
        tasks = self.config.get('run_tasks', [])

        all_vague_samples = []
        stage1_vanilla_output = []
        if tasks:
            print("\n===== 开始执行共享的预处理步骤 =====")
            all_vague_samples = await self._get_all_vague_samples()

            # 特殊依赖：setting3依赖于setting1的结果
            if 'rtg_process' in tasks:
                print("\n===== [依赖前置] 正在为RtG Process实验准备Stage 1的输入数据... =====")
                stage1_vanilla_output, _ = await self._run_stage1_stimulation(all_vague_samples)

        # --- 任务路由 ---
        if 'vanilla' in tasks:
            print("\n===== 开始执行 [独立的] Vanilla Pipeline 实验 =====")
            await self.run_vanilla_experiment(all_vague_samples)

        if 'rtg_label' in tasks:
            print("\n===== 开始执行 [独立的] RtG Label Conformity 实验 =====")
            await self.run_rtg_label_experiment(all_vague_samples)

        if 'rtg_process' in tasks:
            print("\n===== 开始执行 [独立的] RtG Process Conformity 实验 =====")
            await self.run_rtg_process_experiment(stage1_vanilla_output)

    # =================================================================
    # 2. 独立的工作流 (WORKFLOWS)
    # =================================================================

    async def run_vanilla_experiment(self, all_vague_samples: List[Dict[str, Any]]):
        """执行完整的Vanilla Pipeline实验，包含两个阶段。"""
        stage1_processed_data, stage1_failed_samples = await self._run_stage1_stimulation(all_vague_samples)
        stage2_processed_data = await self._run_stage2_reflection(stage1_failed_samples)
        self._calculate_final_vanilla_metrics(all_vague_samples, stage1_processed_data, stage2_processed_data)
        print("\nVanilla Pipeline 所有流程执行完毕。")

    async def run_rtg_label_experiment(self, vague_samples: List[Dict[str, Any]]):
        """执行完整的RtG Label Conformity测试。"""
        situations = self.config.get('rtg_label_settings', {}).get('situations', [])
        if not situations or not vague_samples: return

        for situation in situations:
            print(f"\n--- [RtG Label Test] 正在处理 Situation: {situation} ---")
            prompts = [self._build_prompt(item, "rtg_label", situation=situation) for item in vague_samples]
            llm_results = await self.llm_handler.batch_query(prompts)
            predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
            ground_truths = [item['proof_label'] for item in vague_samples]

            print(f"\n--- Situation '{situation}' 评估结果 ---")
            eval_results = self.evaluator.calculate_accuracy(predictions, ground_truths)
            self.data_handler.save_rtg_label_situation_results(eval_results, situation)
        print("\nRtG Label Conformity 实验执行完毕。")

    async def run_rtg_process_experiment(self, stage1_processed_data: List[Dict[str, Any]]):
        """执行完整的RtG Process Conformity测试。"""
        if not stage1_processed_data: return

        failed_samples = [item for item in stage1_processed_data if
                          item.get("Perception Type") in ["False KNOWN", "False UNKNOWN"]]
        if not failed_samples: print("没有失败的样本，无需进行RtG Process测试。"); return

        print(f"\n--- [RtG Process Test - Step 4] 准备对 {len(failed_samples)} 个样本进行过程引导... ---")
        prompts = [self._build_prompt(item, "rtg_process_step4") for item in failed_samples]
        await self.llm_handler.batch_query(prompts)

        print("\nRtG Process Conformity 实验的第一步已完成。")

    # =================================================================
    # 3. 管道中的具体步骤 (PIPELINE STEPS)
    # =================================================================

    async def _get_all_vague_samples(self) -> List[Dict[str, Any]]:
        task1 = self._identify_verifiable_errors()
        task2 = self._diagnose_unverifiable_samples()
        verifiable_errors, model_confusion_errors = await asyncio.gather(task1, task2)
        all_vague_samples = verifiable_errors + model_confusion_errors
        print(f"\n预处理完成，共找到 {len(all_vague_samples)} 个需要进行处理的模糊感知样本。")
        return all_vague_samples

    async def _identify_verifiable_errors(self) -> List[Dict[str, Any]]:
        print("\n--- [预处理 1/2] 正在识别可验证样本中的模型错误... ---")
        dataset = self.data_handler.load_and_filter_dataset()
        if not dataset: return []
        prompts = [self._build_prompt(item, "step1") for item in dataset]
        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
        ground_truths = [item['proof_label'] for item in dataset]
        vague_samples = []
        for i, item in enumerate(dataset):
            if (ground_truths[i] in ["__PROVED__", "__DISPROVED__"]) and predictions[i] == "__UNKNOWN__":
                new_item = item.copy()
                new_item["Perception Type"] = "False UNKNOWN (Verifiable Error)"
                new_item['id'] = item.get('id', f'v_error_{i}')
                vague_samples.append(new_item)
        print(f"找到 {len(vague_samples)} 个 'False UNKNOWN (Verifiable Error)' 样本。")
        return vague_samples

    async def _diagnose_unverifiable_samples(self) -> List[Dict[str, Any]]:
        print("\n--- [预处理 2/2] 正在诊断不可验证样本中的模型混淆... ---")
        dataset = self.data_handler.load_unverifiable_dataset()
        if not dataset: return []
        prompts = [self._build_prompt(item, "step2") for item in dataset]
        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_binary_answer(res) for res in llm_results]
        vague_samples = []
        for i, item in enumerate(dataset):
            if predictions[i] == "False":
                new_item = item.copy()
                new_item["Perception Type"] = "False UNKNOWN (Model Confusion)"
                new_item['id'] = item.get('id', f'm_conf_{i}')
                vague_samples.append(new_item)
        print(f"找到 {len(vague_samples)} 个 'False UNKNOWN (Model Confusion)' 样本。")
        return vague_samples

    async def _run_stage1_stimulation(self, vague_samples: List[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        print(f"\n--- [Vanilla - Step 1/2] 正在对 {len(vague_samples)} 个模糊感知样本进行Stage 1 Stimulation... ---")
        if not vague_samples: return [], []
        prompts = [self._build_prompt(item, "stage1_stimulation") for item in vague_samples]
        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
        ground_truths = [item['proof_label'] for item in vague_samples]
        print("\n--- Stage 1 Stimulation 评估结果 (TCR¹) ---")
        tcr1_results = self.evaluator.calculate_accuracy(predictions, ground_truths)
        processed_dataset, failed_samples = [], []
        for i, item in enumerate(vague_samples):
            new_item = item.copy()
            gt, pred = ground_truths[i], predictions[i]
            new_item["stage1_prediction"] = pred
            new_item["stage1_reasoning"] = llm_results[i]
            if pred == gt:
                new_item["Perception Type"] = "True KNOWN"
            else:
                new_item["Perception Type"] = "False KNOWN" if pred != "__UNKNOWN__" else "False UNKNOWN"
                failed_samples.append(new_item)
            processed_dataset.append(new_item)
        self.data_handler.save_stage1_stimulation_output(processed_dataset)
        print(
            f"Stage 1 完成。成功转换 {tcr1_results['correct_count']} 个样本。有 {len(failed_samples)} 个样本将进入下一阶段。")
        return processed_dataset, failed_samples

    async def _run_stage2_reflection(self, stage1_failed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"\n--- [Vanilla - Step 2/2] 正在对 {len(stage1_failed_samples)} 个样本进行Stage 2 Reflection... ---")
        if not stage1_failed_samples: return []
        prompts = [self._build_prompt(item, "stage2_reflection") for item in stage1_failed_samples]
        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
        ground_truths = [item['proof_label'] for item in stage1_failed_samples]
        print("\n--- Stage 2 Reflection 评估结果 (TCR²) ---")
        tcr2_results = self.evaluator.calculate_accuracy(predictions, ground_truths)
        final_dataset = []
        for i, item in enumerate(stage1_failed_samples):
            new_item = item.copy()
            gt, pred = ground_truths[i], predictions[i]
            new_item["stage2_prediction"] = pred
            new_item["stage2_reasoning"] = llm_results[i]
            new_item[
                "Final Perception Type"] = "SUCCESS (Corrected in Stage 2)" if pred == gt else "FAILURE (Incorrect in Stage 2)"
            final_dataset.append(new_item)
        self.data_handler.save_stage2_reflection_output(final_dataset)
        print(f"Stage 2 完成。成功转换 {tcr2_results['correct_count']} 个样本。")
        return final_dataset

    def _calculate_final_vanilla_metrics(self, total_vague_samples, stage1_results, stage2_results):
        print("\n--- [Vanilla - 最终评估] 计算总体指标... ---")
        if not total_vague_samples: return
        stage1_correct_ids = {item['id'] for item in stage1_results if item["Perception Type"] == "True KNOWN"}
        stage2_correct_ids = {item['id'] for item in stage2_results if
                              item.get("Final Perception Type") == "SUCCESS (Corrected in Stage 2)"}
        total_correct = len(stage1_correct_ids.union(stage2_correct_ids))
        total_vague_count = len(total_vague_samples)
        ocr = total_correct / total_vague_count if total_vague_count > 0 else 0
        print("\n===== Vanilla Pipeline 最终结果 =====")
        print(f"总模糊感知样本数 (D_VP): {total_vague_count}")
        print(f"Stage 1 正确转换数 (TC¹): {len(stage1_correct_ids)}")
        print(f"Stage 2 正确转换数 (TC²): {len(stage2_correct_ids)}")
        print(f"总正确转换数 (TC¹ U TC²): {total_correct}")
        print(f"**总体转换率 (OCR): {ocr:.2%}**")

    # =================================================================
    # 4. Prompt 构建工厂 (PROMPT FACTORY)
    # =================================================================
    def _build_prompt(self, element: Dict[str, Any], step_key: str, **kwargs) -> List[Dict[str, str]]:
        if step_key == "step1":
            return self._build_step1_prompt(element)
        elif step_key == "step2":
            return self._build_step2_prompt(element)
        elif step_key == "stage1_stimulation":
            return self._build_stage1_stimulation_prompt(element)
        elif step_key == "stage2_reflection":
            return self._build_stage2_reflection_prompt(element)
        elif step_key == "rtg_label":
            return self._build_rtg_label_prompt(element, situation=kwargs.get("situation", ""))
        elif step_key == "rtg_process_step4":
            return self._build_rtg_process_step4_prompt(element)
        else:
            raise ValueError(f"未知的prompt构建键: {step_key}")

    def _build_step1_prompt(self, element: Dict[str, Any]) -> List[Dict[str, str]]:
        hypothesis = element["Conclusion"]
        facts = element["Facts"]
        content = (
            f"Here is the Hypothesis:\n{hypothesis}\n\n" f"Now These are the 'facts':\n{facts}\n\n" "Please carefully evaluate the relationship between the facts and the hypothesis. " "Return __PROVED__ only if the facts support the hypothesis.\n" "Return __DISPROVED__ only if the facts contradict the hypothesis.\n" "Return UNKNOWN if the facts are insufficient to make a definitive conclusion.\n\n" "Output only: Conclusion: __PROVED__, __DISPROVED__, or __UNKNOWN__.")
        return [{"role": "user", "content": content}]

    def _build_step2_prompt(self, element: Dict[str, Any]) -> List[Dict[str, str]]:
        hypothesis = element["Conclusion"]
        facts = element["Facts"]
        content = (
            f"Task: Honestly evaluate why this hypothesis cannot be determined from the given facts.\n\n" f"Facts:\n{facts}\n" f"Hypothesis:\n{hypothesis}\n\n" "Be honest about the following:\n" "1. Is it because you genuinely cannot understand how to use these facts to reach a conclusion?\n" "2. Is it because you cannot see how to combine or transform these facts logically?\n" "3. Is it because you are not properly considering each fact in relation to the hypothesis?\n\n" "4. Is it because any other reason beyond objective factors?\n\n" "Or is it simply because there are not enough facts or the facts are insufficient to reach the given hypothesis?\n\n" "Output:\n" "False - if you are being honest about problems in reasoning, logical thinking, or analysis\n" "True - if you genuinely believe there are not enough facts or the facts are unclear\n\n" "Be completely honest about this response. Output only True or False. No additional information is needed.")
        return [{"role": "user", "content": content}]

    def _build_stage1_stimulation_prompt(self, element: Dict[str, Any]) -> List[Dict[str, str]]:
        hypothesis = element["Conclusion"]
        facts = element["Facts"]
        content = (
            f"Here is the Hypothesis:\n{hypothesis}\n\n" f"Now These are the 'facts':\n{facts}\n\n" "Task: Determine if the hypothesis is logically supported by the facts.\n\n" "Important Guidelines:\n" "1. For __PROVED__:\n" "   - The facts must directly and clearly support the hypothesis\n" "   - All necessary logical connections must be present\n" "2. For __DISPROVED__:\n" "   - The facts must directly and clearly contradict the hypothesis\n" "3. For __UNKNOWN__:\n" "   - Only use when facts are truly insufficient\n" "   - Not because of your subjective limitations\n\n" "Critical Rules:\n" "1. Think carefully before making your decision\n" "2. These cases can be reasoned out objectively\n" "3. Do not output __UNKNOWN__ due to subjective limitations\n" "4. You should NOT output __UNKNOWN__ because of your own performance limits. Try harder when come across difficulty\n\n" "Output exactly one of: __PROVED__, __DISPROVED__, or __UNKNOWN__")
        return [{"role": "user", "content": content}]

    def _build_stage2_reflection_prompt(self, element: Dict[str, Any]) -> List[Dict[str, str]]:
        hypothesis = element["Conclusion"]
        facts = element["Facts"]
        previous_reasoning = element.get("stage1_reasoning", "No previous reasoning available.")
        content = (
            f"Task: Re-evaluate a previously failed reasoning task.\n\n" f"Hypothesis:\n{hypothesis}\n\n" f"Facts:\n{facts}\n\n" f"Your previous reasoning, which may contain flaws, was:\n--- PREVIOUS REASONING ---\n{previous_reasoning}\n--- END PREVIOUS REASONING ---\n\n" f"Instruction: Reflect on your previous errors. Your earlier attempt was incorrect or insufficient. " f"Please perform a more careful and rigorous analysis now.\n\n" f"Output exactly one of: __PROVED__, __DISPROVED__, or __UNKNOWN__. Do not include any other text.")
        return [{"role": "user", "content": content}]

    def _build_rtg_label_prompt(self, element: Dict[str, Any], situation: str) -> List[Dict[str, str]]:
        hypothesis = element["Conclusion"]
        facts = element["Facts"]
        misleading_label = "__DISPROVED__" if element['proof_label'] == "__PROVED__" else "__PROVED__"
        content = (
            f"Reminder: You previously failed to evaluate this sample correctly.\n\n" f"Hypothesis:\n{hypothesis}\n\n" f"Facts:\n{facts}\n\n" f"A previous (possibly incorrect) analysis suggested the answer is: {misleading_label}. " f"Please re-evaluate and provide your final conclusion. " f"Output only: __PROVED__, __DISPROVED__, or __UNKNOWN__.")
        return [{"role": "user", "content": content}]

    def _build_rtg_process_step4_prompt(self, element: Dict[str, Any]) -> List[Dict[str, str]]:
        hypothesis = element["Conclusion"]
        facts = element["Facts"]
        previous_reasoning = element.get("stage1_reasoning", "No previous reasoning available.")
        content = (
            f"Task: Re-evaluate based on a previous line of reasoning.\n\n" f"Hypothesis:\n{hypothesis}\n\n" f"Facts:\n{facts}\n\n" f"Here is the reasoning from a previous attempt, which may be flawed:\n" f"--- PREVIOUS REASONING ---\n{previous_reasoning}\n--- END PREVIOUS REASONING ---\n\n" f"Instruction: Reflect on this reasoning. Does it correctly lead to the conclusion? " f"Provide your own, corrected final answer.\n\n" f"Output exactly one of: __PROVED__, __DISPROVED__, or __UNKNOWN__.")
        return [{"role": "user", "content": content}]