import asyncio
import json
from typing import List, Dict, Any, Tuple

# Import all project dependency modules
from .data_handler import DataHandler
from .llm_handler import LLMHandler
from .evaluator import Evaluator


class WakenllmPipeline:
    """
    Core workflow of the WAKENLLM experimental framework.
    Responsible for orchestrating all steps including data processing, model calls, and evaluation.
    """

    def __init__(self, config: Dict[str, Any], data_handler: DataHandler, llm_handler: LLMHandler,
                 evaluator: Evaluator):
        self.config = config
        self.data_handler = data_handler
        self.llm_handler = llm_handler
        self.evaluator = evaluator
        print("WakenllmPipeline initialization completed.")

    # =================================================================
    # 1. Main Run Entry (RUN METHOD) - Decision Center
    # =================================================================
    async def run(self):
        """
        Route to different experimental workflows based on configuration.
        """
        tasks = self.config.get('run_tasks', [])

        all_vague_samples = []
        stage1_vanilla_output = []
        if tasks:
            print("\n===== Starting shared preprocessing steps =====")
            all_vague_samples = await self._get_all_vague_samples()

            if 'rtg_process' in tasks:
                print("\n===== [Dependency Setup] Preparing Stage 1 input data for RtG Process experiment... =====")
                stage1_vanilla_output, _ = await self._run_stage1_stimulation(all_vague_samples)

        # --- Task Routing ---
        if 'vanilla' in tasks:
            print("\n===== Starting [Independent] Vanilla Pipeline experiment =====")
            await self.run_vanilla_experiment(all_vague_samples)

        if 'rtg_label' in tasks:
            print("\n===== Starting [Independent] RtG Label Conformity experiment =====")
            await self.run_rtg_label_experiment(all_vague_samples)

        if 'rtg_process' in tasks:
            print("\n===== Starting [Independent] RtG Process Conformity experiment =====")
            await self.run_rtg_process_experiment(stage1_vanilla_output)

    # =================================================================
    # 2. Independent Workflows (WORKFLOWS)
    # =================================================================

    async def run_vanilla_experiment(self, all_vague_samples: List[Dict[str, Any]]):
        """Execute complete Vanilla Pipeline experiment, including two stages."""
        stage1_processed_data, stage1_failed_samples = await self._run_stage1_stimulation(all_vague_samples)
        stage2_processed_data = await self._run_stage2_reflection(stage1_failed_samples)
        self._calculate_final_vanilla_metrics(all_vague_samples, stage1_processed_data, stage2_processed_data)
        print("\nVanilla Pipeline all workflows completed.")

    async def run_rtg_label_experiment(self, vague_samples: List[Dict[str, Any]]):
        """Execute complete RtG Label Conformity test."""
        situations = self.config.get('rtg_label_settings', {}).get('situations', [])
        if not situations or not vague_samples: return

        all_accuracies = []
        for situation in situations:
            print(f"\n--- [RtG Label Test] Processing Situation: {situation} ---")
            prompts = [self._build_prompt(item, "rtg_label", situation=situation) for item in vague_samples]
            llm_results = await self.llm_handler.batch_query(prompts)
            predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
            ground_truths = [item['proof_label'] for item in vague_samples]

            print(f"\n--- Situation '{situation}' Evaluation Results ---")
            eval_results = self.evaluator.calculate_accuracy(predictions, ground_truths)
            all_accuracies.append(eval_results['accuracy'])
            self.data_handler.save_rtg_label_situation_results(eval_results, situation)

        avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
        summary = {
            "task": "rtg_label",
            "model": self.config.get("model_name"),
            "dataset": self.config.get("dataset_name"),
            "metrics": {
                "Average_Conformity_Accuracy": avg_accuracy,
                "OCR": avg_accuracy
            }
        }
        summary_path = self.data_handler.results_dir / f"summary_rtg_label_{self.config.get('dataset_name')}_{self.config.get('model_name')}.json"
        self.data_handler.save_json(summary, summary_path)
        print(f"\n✔️  RtG Label Conformity Average Accuracy: {avg_accuracy:.2%}")
        print(f"✔️  Visualization summary saved to: {summary_path}")

        print("\nRtG Label Conformity experiment completed.")

    async def run_rtg_process_experiment(self, all_vague_samples: List[Dict[str, Any]]):
        """Execute complete RtG Process Conformity test, strictly replicating old step4,5,6 logic."""
        if not all_vague_samples:
            print("No vague samples, skipping RtG Process Conformity test.")
            return

        print(
            f"\n--- [RtG Process Test - Step 1/3] Generating initial reasoning for {len(all_vague_samples)} samples... ---")
        step4_prompts = [self._build_prompt(item, "rtg_process_step4") for item in all_vague_samples]
        step4_llm_results = await self.llm_handler.batch_query(step4_prompts)

        reflection1_output = []
        step4_predictions = [self.evaluator.parse_llm_output(res) for res in step4_llm_results]
        ground_truths = [item['proof_label'] for item in all_vague_samples]
        for i, item in enumerate(all_vague_samples):
            new_item = item.copy()
            gt, pred = ground_truths[i], step4_predictions[i]
            new_item["reasoning"] = step4_llm_results[i]
            if gt in ["__PROVED__", "__DISPROVED__"]:
                new_item["Perception Type"] = "True KNOWN" if pred == gt else (
                    "False KNOWN" if pred != "__UNKNOWN__" else "False UNKNOWN")
            else:
                new_item["Perception Type"] = "True UNKNOWN" if pred == "__UNKNOWN__" else "False KNOWN"
            reflection1_output.append(new_item)

        print("Stage 1 complete.")

        step5_input = [item for item in reflection1_output if item["Perception Type"] == "False KNOWN"]
        print(f"\n--- [RtG Process Test - Step 2/3] Reflecting on {len(step5_input)} 'False KNOWN' samples... ---")
        if step5_input:
            step5_prompts = [self._build_prompt(item, "rtg_process_step5_6") for item in step5_input]
            await self.llm_handler.batch_query(step5_prompts)
        print("Stage 2 complete.")

        step6_input = reflection1_output
        print(f"\n--- [RtG Process Test - Step 3/3] Final reflection on all {len(step6_input)} samples... ---")
        step6_prompts = [self._build_prompt(item, "rtg_process_step5_6") for item in step6_input]
        step6_llm_results = await self.llm_handler.batch_query(step6_prompts)

        step6_predictions = [self.evaluator.parse_llm_output(res) for res in step6_llm_results]
        step6_ground_truths = [item['proof_label'] for item in step6_input]
        print("\n--- RtG Process Conformity Final Evaluation ---")
        final_eval_results = self.evaluator.calculate_accuracy(step6_predictions, step6_ground_truths)

        summary = {
            "task": "rtg_process",
            "model": self.config.get("model_name"),
            "dataset": self.config.get("dataset_name"),
            "metrics": {
                "Final_Process_Accuracy": final_eval_results['accuracy'],
                "OCR": final_eval_results['accuracy']
            }
        }
        summary_path = self.data_handler.results_dir / f"summary_rtg_process_{self.config.get('dataset_name')}_{self.config.get('model_name')}.json"
        self.data_handler.save_json(summary, summary_path)
        print(f"✔️  Visualization summary saved to: {summary_path}")
        print("\nRtG Process Conformity experiment completed.")

    # =================================================================
    # 3. Pipeline Steps
    # =================================================================

    async def _get_all_vague_samples(self) -> List[Dict[str, Any]]:
        task1 = self._identify_verifiable_errors()
        task2 = self._diagnose_unverifiable_samples()
        verifiable_errors, model_confusion_errors = await asyncio.gather(task1, task2)
        all_vague_samples = verifiable_errors + model_confusion_errors
        print(f"\nPreprocessing complete. Found {len(all_vague_samples)} vague perception samples.")
        return all_vague_samples

    async def _identify_verifiable_errors(self) -> List[Dict[str, Any]]:
        print("\n--- [Preprocessing 1/2] Identifying model errors on verifiable samples... ---")
        dataset = self.data_handler.load_and_filter_dataset()
        if not dataset: return []

        prompts = []
        for i, item in enumerate(dataset):
            try:
                prompts.append(self._build_prompt(item, "step1"))
            except KeyError as e:
                print(f"\n[!!!] WARNING: Skipping item {i} in verifiable dataset due to missing key: {e}")
                print(f"      Problematic item data: {item}\n")
                continue

        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
        ground_truths = [item.get('proof_label') for item in dataset]
        vague_samples = []
        for i, item in enumerate(dataset):
            if (ground_truths[i] in ["__PROVED__", "__DISPROVED__"]) and predictions[i] == "__UNKNOWN__":
                new_item = item.copy()
                new_item["Perception Type"] = "False UNKNOWN (Verifiable Error)"
                new_item['id'] = item.get('id', f'v_error_{i}')
                vague_samples.append(new_item)
        print(f"Found {len(vague_samples)} 'False UNKNOWN (Verifiable Error)' samples.")
        return vague_samples

    async def _diagnose_unverifiable_samples(self) -> List[Dict[str, Any]]:
        print("\n--- [Preprocessing 2/2] Diagnosing model confusion on unverifiable samples... ---")
        dataset = self.data_handler.load_unverifiable_dataset()
        if not dataset: return []

        prompts = []
        for i, item in enumerate(dataset):
            try:
                prompts.append(self._build_prompt(item, "step2"))
            except KeyError as e:
                print(f"\n[!!!] WARNING: Skipping item {i} in unverifiable dataset due to missing key: {e}")
                print(f"      Problematic item data: {item}\n")
                continue

        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_binary_answer(res) for res in llm_results]
        vague_samples = []
        for i, item in enumerate(dataset):
            if predictions[i] == "False":
                new_item = item.copy()
                new_item["Perception Type"] = "False UNKNOWN (Model Confusion)"
                new_item['id'] = item.get('id', f'm_conf_{i}')
                vague_samples.append(new_item)
        print(f"Found {len(vague_samples)} 'False UNKNOWN (Model Confusion)' samples.")
        return vague_samples

    async def _run_stage1_stimulation(self, vague_samples: List[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        print(f"\n--- [Vanilla - Step 1/2] Running Stage 1 Stimulation on {len(vague_samples)} samples... ---")
        if not vague_samples: return [], []
        prompts = [self._build_prompt(item, "stage1_stimulation") for item in vague_samples]
        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
        ground_truths = [item.get('proof_label') for item in vague_samples]
        print("\n--- Stage 1 Stimulation Evaluation (TCR¹) ---")
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
            f"Stage 1 complete. Converted {tcr1_results['correct_count']} samples. {len(failed_samples)} samples failed.")
        return processed_dataset, failed_samples

    async def _run_stage2_reflection(self, stage1_failed_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"\n--- [Vanilla - Step 2/2] Running Stage 2 Reflection on {len(stage1_failed_samples)} samples... ---")
        if not stage1_failed_samples: return []
        prompts = [self._build_prompt(item, "stage2_reflection") for item in stage1_failed_samples]
        llm_results = await self.llm_handler.batch_query(prompts)
        predictions = [self.evaluator.parse_llm_output(res) for res in llm_results]
        ground_truths = [item.get('proof_label') for item in stage1_failed_samples]
        print("\n--- Stage 2 Reflection Evaluation (TCR²) ---")
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
        print(f"Stage 2 complete. Converted {tcr2_results['correct_count']} samples.")
        return final_dataset

    def _calculate_final_vanilla_metrics(self, total_vague_samples, stage1_results, stage2_results):
        print("\n--- [Vanilla - Final Evaluation] Calculating overall metrics... ---")
        if not total_vague_samples: return

        stage1_correct_ids = {item.get('id') for item in stage1_results if item.get("Perception Type") == "True KNOWN"}
        stage2_correct_ids = {item.get('id') for item in stage2_results if
                              item.get("Final Perception Type") == "SUCCESS (Corrected in Stage 2)"}

        total_correct = len(stage1_correct_ids.union(stage2_correct_ids))
        total_vague_count = len(total_vague_samples)

        stage1_failed_samples_count = total_vague_count - len(stage1_correct_ids)

        ocr = total_correct / total_vague_count if total_vague_count > 0 else 0
        tcr1 = len(stage1_correct_ids) / total_vague_count if total_vague_count > 0 else 0
        tcr2 = len(stage2_correct_ids) / stage1_failed_samples_count if stage1_failed_samples_count > 0 else 0

        print("\n===== Vanilla Pipeline Final Results =====")
        print(f"Total Vague Samples (D_VP): {total_vague_count}")
        print(f"Stage 1 True Conversions (TC¹): {len(stage1_correct_ids)}")
        print(f"Stage 2 True Conversions (TC²): {len(stage2_correct_ids)}")
        print(f"Overall True Conversions (TC¹ U TC²): {total_correct}")
        print(f"**Overall Conversion Rate (OCR): {ocr:.2%}**")

        summary = {
            "task": "vanilla",
            "model": self.config.get("model_name"),
            "dataset": self.config.get("dataset_name"),
            "metrics": {
                "TCR1": tcr1,
                "TCR2": tcr2,
                "OCR": ocr
            },
            "counts": {
                "total_vague_samples": total_vague_count,
                "stage1_correct": len(stage1_correct_ids),
                "stage2_correct": len(stage2_correct_ids)
            }
        }
        summary_path = self.data_handler.results_dir / f"summary_vanilla_{self.config.get('dataset_name')}_{self.config.get('model_name')}.json"
        self.data_handler.save_json(summary, summary_path)
        print(f"✔️  Visualization summary saved to: {summary_path}")

    # =================================================================
    # 4. Prompt Factory
    # =================================================================
    def _build_prompt(self, element: Dict[str, Any], step_key: str, **kwargs) -> List[Dict[str, str]]:
        # Using .get() for robust data access is a good practice.
        hypothesis = element.get("Conclusion", "")
        facts = element.get("Facts", "")
        proof_label = element.get("proof_label", "")
        stage1_reasoning = element.get("stage1_reasoning", "No previous reasoning available.")
        reasoning = element.get("reasoning", "No previous reasoning available.")

        if step_key == "step1":
            return self._build_step1_prompt(hypothesis, facts)
        elif step_key == "step2":
            return self._build_step2_prompt(hypothesis, facts)
        elif step_key == "stage1_stimulation":
            return self._build_stage1_stimulation_prompt(hypothesis, facts)
        elif step_key == "stage2_reflection":
            return self._build_stage2_reflection_prompt(hypothesis, facts, stage1_reasoning)
        elif step_key == "rtg_label":
            return self._build_rtg_label_prompt(hypothesis, facts, proof_label, situation=kwargs.get("situation", ""))
        elif step_key == "rtg_process_step4":
            return self._build_rtg_process_step4_prompt(hypothesis, facts)
        elif step_key == "rtg_process_step5_6":
            return self._build_rtg_process_step5_6_prompt(hypothesis, facts, reasoning)
        else:
            raise ValueError(f"Unknown prompt build key: {step_key}")

    def _build_step1_prompt(self, hypothesis: str, facts: str) -> List[Dict[str, str]]:
        content = (
            f"Here is the Hypothesis:\n{hypothesis}\n"
            f"Now These are the 'facts':\n{facts}\n"
            "Please carefully evaluate the relationship between the facts and the hypothesis. Return UNKNOWN if the facts are insufficient to make a definitive conclusion.\n"
            "Return __PROVED__ only if the facts support the hypothesis.\n"
            "Return __DISPROVED__ only if the facts contradict the hypothesis.\n"
            "Output only: Conclusion: __PROVED__, __DISPROVED__, or __UNKNOWN__."
        )
        return [{"role": "user", "content": content}]

    def _build_step2_prompt(self, hypothesis: str, facts: str) -> List[Dict[str, str]]:
        content = (
            f"Task: Honestly evaluate why this hypothesis cannot be determined from the given facts.\n\n" f"Facts:\n{facts}\n" f"Hypothesis:\n{hypothesis}\n\n" "Be honest about the following:\n" "1. Is it because you genuinely cannot understand how to use these facts to reach a conclusion?\n" "2. Is it because you cannot see how to combine or transform these facts logically?\n" "3. Is it because you are not properly considering each fact in relation to the hypothesis?\n\n" "4. Is it because any other reason beyond objective factors?\n\n" "Or is it simply because there are not enough facts or the facts are insufficient to reach the given hypothesis?\n\n" "Output:\n" "False - if you are being honest about problems in reasoning, logical thinking, or analysis\n" "True - if you genuinely believe there are not enough facts or the facts are unclear\n\n" "Be completely honest about this response. Output only True or False. No additional information is needed.")
        return [{"role": "user", "content": content}]

    def _build_stage1_stimulation_prompt(self, hypothesis: str, facts: str) -> List[Dict[str, str]]:
        content = (
            f"Here is the Hypothesis:\n{hypothesis}\n\n"
            f"Now These are the 'facts':\n{facts}\n\n"
            "Task: Determine if the hypothesis is logically supported by the facts.\n\n"
            "Important Guidelines:\n"
            "1. For __PROVED__:\n"
            "   - The facts must directly and clearly support the hypothesis\n"
            "   - All necessary logical connections must be present\n"
            "   - No contradictory facts should exist\n"
            "   - The support must be unambiguous\n\n"
            "2. For __DISPROVED__:\n"
            "   - The facts must directly and clearly contradict the hypothesis\n"
            "   - The contradiction must be explicit and unambiguous\n"
            "   - No supporting facts should exist\n"
            "   - The contradiction must be logically sound\n\n"
            "3. For __UNKNOWN__:\n"
            "   - Only use when facts are truly insufficient\n"
            "   - Not because of your subjective limitations\n"
            "   - Not because of unclear reasoning\n"
            "   - Only when missing critical information\n\n"
            "Critical Rules:\n"
            "1. Think carefully before making your decision\n"
            "2. These cases can be reasoned out objectively\n"
            "3. Do not output __UNKNOWN__ due to subjective limitations\n"
            "4. Output exactly one of: __PROVED__, __DISPROVED__, or __UNKNOWN__\n"
            "5. Do not include any additional text or explanation\n\n"
            "6. You should NOT output __UNKNOWN__ because of your own performance limits. Try harder when come across difficulty\n"
            "Remember: Your output must be exactly one of these three options:\n"
            "__PROVED__\n"
            "__DISPROVED__\n"
            "__UNKNOWN__"
        )
        return [{"role": "user", "content": content}]

    def _build_stage2_reflection_prompt(self, hypothesis: str, facts: str, previous_reasoning: str) -> List[
        Dict[str, str]]:
        content = (
            f"Task: Re-evaluate a previously failed reasoning task.\n\n" f"Hypothesis:\n{hypothesis}\n\n" f"Facts:\n{facts}\n\n" f"Your previous reasoning, which may contain flaws, was:\n--- PREVIOUS REASONING ---\n{previous_reasoning}\n--- END PREVIOUS REASONING ---\n\n" f"Instruction: Reflect on your previous errors. Your earlier attempt was incorrect or insufficient. " f"Please perform a more careful and rigorous analysis now.\n\n" f"Output exactly one of: __PROVED__, __DISPROVED__, or __UNKNOWN__. Do not include any other text.")
        return [{"role": "user", "content": content}]

    def _build_rtg_label_prompt(self, hypothesis: str, facts: str, proof_label: str, situation: str) -> List[
        Dict[str, str]]:
        misleading_label = "__DISPROVED__" if proof_label == "__PROVED__" else "__PROVED__"
        content = (
            f"Reminder: You previously failed to evaluate this sample correctly.\n\n" f"Hypothesis:\n{hypothesis}\n\n" f"Facts:\n{facts}\n\n" f"A previous (possibly incorrect) analysis suggested the answer is: {misleading_label}. " f"Please re-evaluate and provide your final conclusion. " f"Output only: __PROVED__, __DISPROVED__, or __UNKNOWN__.")
        return [{"role": "user", "content": content}]

    def _build_rtg_process_step4_prompt(self, hypothesis: str, facts: str) -> List[Dict[str, str]]:
        content = (
            f"Here is the Hypothesis:\n{hypothesis}\n"
            f"Now These are the 'facts':\n{facts}\n"
            "The Data were you previously mis-inferenced, pay more attention and not to make mistakes. "
            "Think twice before outputing, as they can be reasoned out because of objective factors. "
            "Subjective ability limits should be overcomed.\n\n"
            "Please follow these steps:\n"
            "1. First, carefully analyze the facts and the hypothesis\n"
            "2. Then, explain your reasoning process step by step\n"
            "3. Finally, conclude with your answer\n\n"
            "Important guidelines:\n"
            "1. If the facts logically support the hypothesis, output __PROVED__\n"
            "2. If the facts logically contradict the the hypothesis , output __DISPROVED__\n"
            "3. If the facts are insufficient to prove or disprove the answer to the Hypothesis, output __UNKNOWN__\n"
            "4. Think twice before you output __UNKNOWN__, it is prohibited to output __UNKNOWN__ because of subjective factors like your ability\n\n"
            "Please format your response as follows:\n"
            "Reasoning Process:\n"
            "[Your step-by-step reasoning here]\n\n"
            "Conclusion: [__PROVED__ or __DISPROVED__ or __UNKNOWN__]"
        )
        return [{"role": "user", "content": content}]

    def _build_rtg_process_step5_6_prompt(self, hypothesis: str, facts: str, previous_reasoning: str) -> List[
        Dict[str, str]]:
        content = (
            f"Here is the Hypothesis:\n{hypothesis}\n"
            f"Now These are the 'facts':\n{facts}\n"
            "You previously mis inferenced the label, it coule be worked out by the facts, but you worked out the wrong label. Now your task is to determine if the hypothesis is PROVED or DISPROVED or UNKNOWNby the facts.\n"
            "Important guidelines:\n"
            "1. If the facts logically support the Hypothesis, output __PROVED__\n"
            "2. If the facts logically contradict the Hypothesis, output __DISPROVED__\n"
            "3. If the facts are insufficient to prove or disprove the Hypothesis, output __UNKNOWN__\n"
            "3. Do not output any additional text, only the label (__PROVED__, __DISPROVED__, __UNKNOWN__)\n\n"
            "4. Think twice before you output the __UNKNOWN__, it is prohibited to output __UNKNOWN__ because of subjective factos like your ability.\n"
            "Previous reasoning process are here, learn from them and do not make the same mistake:\n"
            f"{previous_reasoning}\n"
        )
        return [{"role": "user", "content": content}]