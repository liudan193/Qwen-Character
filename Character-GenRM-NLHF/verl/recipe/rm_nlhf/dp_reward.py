"""
Implement a multiprocess PPOCritic for SequenceClassification models
"""

import itertools
import logging
import os
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
import torch.nn.functional as F
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.models import transformers
from verl.protocol import pad_dataproto_to_divisor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOReward():
    def __init__(self, config, model_config, reward_module: nn.Module, tokenizer, reward_optimizer: optim.Optimizer):
        self.config = config
        self.model_config = model_config
        self.reward_module = reward_module
        self.tokenizer = tokenizer
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        assert self.use_remove_padding == False
        print(f"Reward use_remove_padding={self.use_remove_padding}")

        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch):
        """
        Perform forward pass using AutoModelForSequenceClassification.
        Return predicted scores for each sample.
        Pad sequences to the actual maximum length within the mini-batch.
        """
        # Extract input tensors from the batch
        input_ids_batch = micro_batch["input_ids"]
        attention_mask_batch = micro_batch["attention_mask"]
        position_ids_batch = micro_batch["position_ids"]

        batch_size = input_ids_batch.size(0)

        # 1. Find the actual maximum sequence length within the batch
        actual_lengths = attention_mask_batch.sum(dim=1)  # shape: [batch_size]
        max_actual_len = int(actual_lengths.max().item())

        # 2. Slice all samples according to the maximum length (for left padding)
        # Assuming original seq_len = 512 and max_actual_len = 300, take [:, -300:]
        trimmed_input_ids = input_ids_batch[:, -max_actual_len:]  # [batch_size, max_actual_len]
        trimmed_attention_mask = attention_mask_batch[:, -max_actual_len:]  # [batch_size, max_actual_len]
        trimmed_position_ids = position_ids_batch[:, -max_actual_len:]  # [batch_size, max_actual_len]

        # 3. Perform forward pass with autocast
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.reward_module(
                input_ids=trimmed_input_ids,
                attention_mask=trimmed_attention_mask,
                position_ids=trimmed_position_ids,
                use_cache=False,
            )

        # 4. SequenceClassification model outputs logits of shape [batch_size, num_labels]
        # Since num_labels=1, directly extract the scores
        final_scores = output.logits.squeeze(-1)  # shape: [batch_size]

        # print("=" * 50)
        # print("max_actual_len:", max_actual_len)
        # print("final_scores shape:", final_scores.shape)
        # print("final_scores:", final_scores)
        # print("=" * 50)

        return final_scores

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.reward_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.reward_optimizer.zero_grad()
        else:
            self.reward_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp reward", logger=logger)
    def compute_rewards(self, data: DataProto) -> torch.Tensor:
        """
        Compute reward scores.
        Returns shape: (bs,)
        """
        self.reward_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)

        reward_lst = []
        for idx, micro_batch in enumerate(tqdm(micro_batches, desc='Calculate Reward Score')):
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                score = self._forward_micro_batch(micro_batch)
            reward_lst.append(score)

        rewards = torch.cat(reward_lst, dim=0)  # shape: (bs,)

        # Do not enable this
        # Apply sigmoid to map scores to [0, 1]
        # rewards = torch.sigmoid(rewards)

        # Do not enable this
        # if self.config.reward_scale:  # scale the reward to 0 or 1
        #     rewards = torch.sigmoid(rewards)
        #     rewards = (rewards > 0.5).float()

        return rewards

    @GPUMemoryLogger(role="dp reward", logger=logger)
    def update_reward(self, data: DataProto):
        """
        Update the reward model.
        Uses MSE loss (or BCE loss if labels are binary 0/1).
        """

        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        select_keys = [
            "input_ids", "attention_mask", "position_ids", "labels"
        ]

        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        pbar = tqdm(desc='Update Reward Model', total=len(dataloader) * self.config.ppo_epochs)

        # Define loss function
        # Use MSE if labels are continuous values
        loss_fct = nn.MSELoss()
        # For binary classification labels (0 or 1), BCE can be used instead
        # loss_fct = nn.BCEWithLogitsLoss()

        for epoch in range(self.config.ppo_epochs):
            # print("for epoch in range(self.config.ppo_epochs):")
            for batch_idx, data in enumerate(dataloader):
                # print("for batch_idx, data in enumerate(dataloader):")
                # Split mini-batch into micro-batches for gradient accumulation
                batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                self.reward_optimizer.zero_grad()

                for mini_batch in batches:
                    # print("for mini_batch in batches:")
                    # ------------------- Truncation logic for left padding -------------------
                    # print("# ------------------- Truncation logic for left padding -------------------")
                    attention_mask = mini_batch["attention_mask"].to(get_device_id())

                    # 1. Compute the actual maximum sequence length in the micro-batch
                    max_len_in_batch = int(attention_mask.sum(dim=1).max().item())

                    # 2. Truncate all sequence-related tensors from the end
                    input_ids = mini_batch["input_ids"][:, -max_len_in_batch:].to(get_device_id())
                    attention_mask = attention_mask[:, -max_len_in_batch:]
                    position_ids = mini_batch["position_ids"][:, -max_len_in_batch:].to(get_device_id())

                    # labels is a scalar array with shape: [batch_size]
                    labels = mini_batch["labels"].to(get_device_id())
                    if labels.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        labels = labels.float()
                    # ------------------- End of truncation logic -------------------

                    # print("with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):")
                    with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                        # Perform forward pass using truncated tensors
                        outputs = self.reward_module(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )
                        # print("Done: outputs = self.reward_module()")

                        # SequenceClassification model outputs logits of shape [batch_size, num_labels]
                        # Since num_labels=1, squeeze the last dimension
                        predictions = outputs.logits.squeeze(-1)  # shape: [batch_size]

                        # Compute loss
                        # MSE loss (for continuous labels)
                        # print("Start: loss = loss_fct(predictions, labels)")
                        loss = loss_fct(predictions, labels)
                        # print("Done: loss = loss_fct(predictions, labels)")

                        # For BCEWithLogitsLoss (for binary classification)
                        # loss = loss_fct(predictions, labels)

                    # print("Start: loss.backward()")
                    loss = loss / self.gradient_accumulation
                    loss.backward()
                    # print("Done: loss.backward()")

                    loss_item = loss.item() * self.gradient_accumulation

                    data_dict = {"reward/loss": loss_item}
                    append_to_dict(metrics, data_dict)

                # print("Start: torch.cuda.empty_cache()")
                # Try to clean GPU memory
                del outputs, predictions, loss, input_ids, attention_mask, position_ids, labels
                torch.cuda.empty_cache()
                # print("Done: torch.cuda.empty_cache()")

                pbar.update(1)
                grad_norm = self._optimizer_step()
                data_dict = {"reward/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data_dict)

        pbar.close()

        # print("Update Reward Done（inside function）.")

        final_metrics = {key: sum(val) / len(val) for key, val in metrics.items()}
        return final_metrics