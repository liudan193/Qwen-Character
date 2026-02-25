import re
from collections import defaultdict
import numpy as np
from verl import DataProto
import asyncio
import torch
import time
from tqdm import tqdm
from .reward_function import parse_llm_evaluation


class CollaborateDataGenerator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def preprocess_reward_data(self, data: DataProto, data_filter: bool=False) -> DataProto:
        """
        Preprocess reward data for a batch of data.
        Args:
            data: DataProto object containing the input data.
            data_filter: Whether to data_filter. If `data_filter=True`, data filtering is required. If False, all data should be preserved.
        """

        max_prompt_length = self.config.get("max_prompt_length", 6144)
        max_response_length = self.config.get("max_response_length", 4096)

        # Obtain appropriate training data:
        #   (1) Extract the training prompts used by MetaRM;
        #   (2) Extract the training scores used by MetaRM;

        # Skip data filtering
        new_orig_batch = dict(data.batch)
        new_orig_non_tensor_batch = dict(data.non_tensor_batch)

        # Construct SFT batch
        extra_infos = data.non_tensor_batch.get("extra_info", None)
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        prompt_ids_list = []
        labels_list = []
        raw_prompt = []
        for i, extra_info in enumerate(extra_infos):
            # Construct data

            # Construct input
            conv_his = ""
            for message in extra_info['context']:
                conv_his += f"{message['role']}:\n{message['content']}\n"
            prompt_text = f"""<Conversation History>
{conv_his}
</Conversation History>

<Response A>
{extra_info['response_A']}
</Response A>

<Response B>
{extra_info['response_B']}
</Response B>"""

            # Extract and decode response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length - 1]  # we need to exclude the eos token
            response_text = self.tokenizer.decode(valid_response_ids)

            results = parse_llm_evaluation(response_text)
            critics = f"""<critics>
{results["critics"]}
</critics>"""
            input_ques_wc = (
                f"<conv_his>{prompt_text}</conv_his>\n\n"
                f"<critics>{critics}</critics>"
            )
            label = data.non_tensor_batch["weighted_sum_score"][i]

            # Construct tokenized input
            # Generate prompt text (containing only user question, with generation marker added)
            prompt_inputs = self.tokenizer(
                input_ques_wc, max_length=max_prompt_length + max_response_length,
                return_tensors="pt", add_special_tokens=False,
                truncation=True, padding_side="left", padding="max_length"
            )
            prompt_input_ids = prompt_inputs["input_ids"]
            prompt_attention_mask = prompt_inputs["attention_mask"]

            # ===== Concatenate into model input =====
            position_ids = self.compute_position_id_from_first_valid_1d(prompt_attention_mask).unsqueeze(0)

            # Final data content
            input_ids_list.append(prompt_input_ids)
            attention_mask_list.append(prompt_attention_mask)
            position_ids_list.append(position_ids)
            prompt_ids_list.append(prompt_input_ids)
            labels_list.append(label)
            raw_prompt.append(input_ques_wc)  # Save raw input; here raw_prompt is just a str

        # ===== Concatenate into batched tensors =====
        new_batch = {}
        new_non_tensor_batch = {}
        new_batch["input_ids"] = torch.cat(input_ids_list, dim=0)
        new_batch["attention_mask"] = torch.cat(attention_mask_list, dim=0)
        new_batch["position_ids"] = torch.cat(position_ids_list, dim=0)
        new_batch["prompts"] = torch.cat(prompt_ids_list, dim=0)
        new_batch["labels"] = torch.tensor(labels_list, dtype=torch.float32)
        new_non_tensor_batch["raw_prompt"] = np.array(raw_prompt, dtype=object)

        # Generate DataProto
        # Merge fields
        merged_batch = dict(new_orig_batch)
        merged_batch.update(new_batch)
        merged_non_tensor_batch = dict(new_orig_non_tensor_batch)
        merged_non_tensor_batch.update(new_non_tensor_batch)
        meta_info = data.meta_info.copy() if data.meta_info else {}

        sft_data = DataProto.from_dict(
            tensors=merged_batch,
            non_tensors=merged_non_tensor_batch,
            meta_info=meta_info
        )

        return sft_data


    def compute_position_id_from_first_valid_1d(self, mask):
        """
        Compute position ids based on the first valid token in a 1D mask.
        """
        # Ensure mask is a 1D tensor
        if mask.dim() != 1:
            mask = mask.squeeze()
            if mask.dim() != 1:
                raise ValueError(f"Input mask must be a 1D tensor, but got shape {mask.shape}")

        if mask.sum() == 0:
            return torch.zeros_like(mask, dtype=torch.long)

        seq_len = mask.shape[0]

        first_valid_pos = torch.argmax(mask.float())

        position_ids = torch.arange(seq_len, device=mask.device, dtype=torch.long) - first_valid_pos
        # Ensure position ids are non-negative
        position_ids = torch.clamp(position_ids, min=0)

        # This step corrects position_ids in right-padding regions
        position_ids.masked_fill_(mask == 0, 0)

        return position_ids