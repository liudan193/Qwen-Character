# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def process_metarm_score(score):
    result = max(0, score-1)
    return result

@register("rm_nlhf")
class RmNlhfRewardManager:
    """The reward manager with batch processing support."""

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", weight_rule=1.0, weight_model=0.5) -> None:
        """
        Initialize the CombineModelRuleV2RewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: Number of decoded responses per batch to print to console for debugging purposes.
            compute_score: Function used to compute reward scores. If None, `default_compute_score` will be used.
            reward_fn_key: Key used to access the data source in non-tensor batch data. Defaults to "data_source".
            weight_rule: Weight for rule-based rewards.
            weight_model: Weight for model-based rewards (i.e., metarm).
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.weight_rule = weight_rule
        self.weight_model = weight_model

    def __call__(self, data: DataProto, return_dict=False):
        # If rm_scores already exist, directly use the existing scores
        rm_scores_tensor = data.batch.get("rm_scores", None)

        # if rm_scores_tensor is not None:
        if "weighted_sum_score" in data.non_tensor_batch.keys():
            # Scores already exist; no need to recompute
            rm_scores = rm_scores_tensor.tolist()
            attention_mask = data.batch["attention_mask"]
            prompt_ids = data.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)
            already_printed = {}  # To control printing frequency

            for i in range(len(data)):
                reward_quant_metarm_score = process_metarm_score(rm_scores[i])

                if data.non_tensor_batch["reward_format"][i] == -1.0:
                    # Fix format errors
                    final_reward = data.non_tensor_batch["reward_format"][i]
                else:
                    if data.non_tensor_batch["extra_info"][i]["dataset_type"] == "Helpsteer3_A":
                        # final_reward = data.non_tensor_batch["weighted_sum_score"][i]
                        final_reward = (
                            self.weight_rule * data.non_tensor_batch["reward_outcome"][i]
                            + self.weight_model * data.non_tensor_batch["reward_quant_critique_score"][i]
                        )
                    else:
                        if data.non_tensor_batch["reward_outcome"][i] == 0.0:
                            final_reward = 0.0
                        else:
                            final_reward = (
                                self.weight_rule * data.non_tensor_batch["reward_outcome"][i]
                                + self.weight_model * reward_quant_metarm_score
                            )

                # Record extra info and convert to float type
                reward_extra_info["weighted_sum_score"].append(float(data.non_tensor_batch["weighted_sum_score"][i]))
                reward_extra_info["reward_format"].append(float(data.non_tensor_batch["reward_format"][i]))
                reward_extra_info["reward_outcome"].append(float(data.non_tensor_batch["reward_outcome"][i]))
                reward_extra_info["reward_critique_score"].append(float(data.non_tensor_batch["reward_critique_score"][i]))
                reward_extra_info["reward_quant_critique_score"].append(float(data.non_tensor_batch["reward_quant_critique_score"][i]))
                reward_extra_info["critique_precision"].append(float(data.non_tensor_batch["critique_precision"][i]))
                reward_extra_info["critique_recall"].append(float(data.non_tensor_batch["critique_recall"][i]))
                reward_extra_info["metarm_score"].append(float(rm_scores[i]))
                reward_extra_info["quant_metarm_score"].append(float(reward_quant_metarm_score))  # Use quantized value
                reward_extra_info["score"].append(float(final_reward))

                # Populate reward tensor (assign value only at the last valid token of the response)
                valid_response_len = valid_response_lengths[i].item()
                reward_tensor[i, valid_response_len - 1] = float(final_reward)

                # Debug printing (consistent with the re-scoring branch)
                data_source = data.non_tensor_batch[self.reward_fn_key][i]
                if already_printed.get(data_source, 0) < self.num_examine:
                    response_str = self.tokenizer.decode(
                        data.batch["responses"][i][:valid_response_len], skip_special_tokens=True
                    )
                    prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                    ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", final_reward)
                    already_printed[data_source] = already_printed.get(data_source, 0) + 1

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor

        else:
            # No existing scores; need to recompute
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)
            prompt_ids = data.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attention_mask = data.batch["attention_mask"]
            valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

            # Decode responses
            responses_str = []
            for i in range(len(data)):
                valid_len = valid_response_lengths[i]
                valid_response_ids = data.batch["responses"][i][:valid_len]
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                responses_str.append(response_str)

            # Extract non-tensor metadata
            ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
            data_sources = data.non_tensor_batch[self.reward_fn_key]
            extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

            # Compute scores
            scores = self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extras,
            )

            rewards = []
            already_printed = {}

            for i in range(len(data)):
                length = valid_response_lengths[i].item()
                score = scores[i]

                if isinstance(score, dict):
                    reward = score["score"]
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                rewards.append(reward)
                reward_tensor[i, length - 1] = reward

                # Debug printing
                data_source = data_sources[i]
                if already_printed.get(data_source, 0) < self.num_examine:
                    response_str = self.tokenizer.decode(
                        data.batch["responses"][i][:length], skip_special_tokens=True
                    )
                    prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                    ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", scores[i])
                    already_printed[data_source] = already_printed.get(data_source, 0) + 1

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor