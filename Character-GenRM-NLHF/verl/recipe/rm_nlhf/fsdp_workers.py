"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
from typing import Union
from openai import OpenAI
import numpy as np
import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.fsdp_workers import RewardModelWorker as verl_RewardModelWorker
from verl.utils.device import get_device_id
from verl.protocol import pad_dataproto_to_divisor
from .dp_reward import DataParallelPPOReward

from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from .collaborate import CollaborateDataGenerator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def filter_helpsteer3_a(data: DataProto) -> DataProto:
    """
    Filter out data where dataset_type is Helpsteer3_A
    """
    extra_infos = data.non_tensor_batch.get("extra_info", None)

    if extra_infos is None:
        raise ValueError("extra_info is None")

    # Create a list of indices to keep
    keep_indices = []

    for i, extra_info in enumerate(extra_infos):
        if extra_info.get("dataset_type", None) == "Helpsteer3_A" and data.non_tensor_batch["reward_format"][i] != -1.0:
            keep_indices.append(i)

    if len(keep_indices) == 0:
        # No matching data, return None
        return None

    # Filter using indices
    keep_indices_tensor = torch.tensor(keep_indices)
    filtered_batch = data.batch[keep_indices_tensor]

    # Filter each key in non_tensor_batch
    filtered_non_tensor_batch = {}
    for key in data.non_tensor_batch.keys():
        filtered_values = [data.non_tensor_batch[key][i] for i in keep_indices]
        filtered_non_tensor_batch[key] = np.array(filtered_values, dtype=object)

    # Return filtered DataProto
    return DataProto(
        batch=filtered_batch,
        non_tensor_batch=filtered_non_tensor_batch,
        meta_info=data.meta_info
    )


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(verl_RewardModelWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__(config)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"

    def _build_reward_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get("trust_remote_code", False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_module = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
                num_labels=1,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

            if config.model.get("enable_gradient_checkpointing", False):
                print("enable gradient checkpointing = False")
                reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self.rank == 0:
            print_model_size(reward_module)

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        log_gpu_memory_usage("Before Reward FSDP", logger=None)
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        print(reward_module.device)
        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(reward_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After reward FSDP", logger=None)

        reward_optimizer = optim.AdamW(
            reward_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
        if warmup_style == "constant":
            reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=reward_optimizer,
                                                                    num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":  # cosine decay
            reward_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=reward_optimizer,
                                                                  num_warmup_steps=num_warmup_steps,
                                                                  num_training_steps=total_steps)

        return model_config, reward_module, reward_optimizer, reward_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.model_config, self.reward_module, self.reward_optimizer, self.reward_lr_scheduler = (
            self._build_reward_model_optimizer(config=self.config))

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.reward = DataParallelPPOReward(
            config=self.config,
            model_config=self.model_config,
            reward_module=self.reward_module,
            tokenizer=self.tokenizer,
            reward_optimizer=self.reward_optimizer
        )

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.reward_module,
            optimizer=self.reward_optimizer,
            lr_scheduler=self.reward_lr_scheduler,
            processing_class=self.input_tokenizer if self._do_switch_chat_template else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

        self.collaborate = CollaborateDataGenerator(self.config.collaborate, self.tokenizer)

    def _get_data_from_other_rank(self, data: DataProto):
        """
        Processes without data receive data from processes that have data.
        Processes with data remain unchanged.

        Args:
            data: Data on the current process (may be None or empty)

        Returns:
            Processed data (processes without data receive broadcasted data; processes with data retain their original data)
        """
        # 1. Determine whether the current process has data
        has_data = False
        current_batch_size = 0
        if data is not None:
            input_ids = data.batch.get("input_ids", None)
            has_data = (input_ids is not None and len(input_ids) > 0)
            if has_data:
                current_batch_size = len(input_ids)

        # 2. Gather status and batch size from all processes
        has_data_tensor = torch.tensor([1 if has_data else 0], dtype=torch.long, device=get_device_id())
        batch_size_tensor = torch.tensor([current_batch_size], dtype=torch.long, device=get_device_id())

        # Create tensors to store each process's status and batch size
        all_has_data = [torch.zeros(1, dtype=torch.long, device=get_device_id()) for _ in range(self.world_size)]
        all_batch_sizes = [torch.zeros(1, dtype=torch.long, device=get_device_id()) for _ in range(self.world_size)]

        torch.distributed.all_gather(all_has_data, has_data_tensor)
        torch.distributed.all_gather(all_batch_sizes, batch_size_tensor)

        # 3. Count how many processes have data and find the maximum batch size
        num_processes_with_data = sum(status.item() for status in all_has_data)
        max_batch_size = max(bs.item() for bs in all_batch_sizes)

        # 4. If no process has data, return immediately
        if num_processes_with_data == 0:
            if self.rank == 0:
                print("No process has data")
            return None

        if self.rank == 0:
            print(f"{num_processes_with_data}/{self.world_size} processes have data, max_batch_size={max_batch_size}")

        # 🔥 5. Processes with data first pad their data to max_batch_size
        if has_data:
            data, pad_size = pad_dataproto_to_divisor(data, max_batch_size)
            if self.rank == 0 and pad_size > 0:
                print(f"Rank {self.rank}: Padded {pad_size} samples from {current_batch_size} to {current_batch_size + pad_size})")

        # 6. If all processes already have data, no broadcasting is needed
        if num_processes_with_data == self.world_size:
            if self.rank == 0:
                print(f"All {self.world_size} processes have data after padding")
                batch_sizes_str = ", ".join([f"rank{i}:{bs.item()}" for i, bs in enumerate(all_batch_sizes)])
                print(f"Batch sizes: {batch_sizes_str}")
            return data

        # 7. Find the first process with data to serve as the broadcast source
        source_rank = -1
        for rank, status in enumerate(all_has_data):
            if status.item() == 1:
                source_rank = rank
                break

        if self.rank == 0:
            print(f"Source rank with data: {source_rank}")

        # 🔥 8. Broadcast the already padded data (all processes must participate)
        if not has_data:
            # Processes without data: receive broadcast
            broadcast_list = [None]
            torch.distributed.broadcast_object_list(
                broadcast_list,
                src=source_rank,
                device=torch.device(get_device_id())
            )
            received_data = broadcast_list[0]

            if self.rank == 0:
                print(f"Rank {self.rank} received padded data from rank {source_rank}, "
                      f"length = {len(received_data.batch['input_ids'])}")

            return received_data
        else:
            # Processes with data must also participate in the broadcast call
            if self.rank == source_rank:
                # source_rank sends its already padded data
                broadcast_list = [data]
            else:
                # Other processes with data still need to call broadcast
                broadcast_list = [None]

            torch.distributed.broadcast_object_list(
                broadcast_list,
                src=source_rank,
                device=torch.device(get_device_id())
            )

            # Return its own already padded data
            return data

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_reward(self, data: DataProto):
        """Use data broadcasting to ensure all processes participate in communication"""

        # 1. Preprocess data
        sft_data = self.collaborate.preprocess_reward_data(data, data_filter=False)
        sft_data = filter_helpsteer3_a(sft_data)

        # 2. Check if the current process has valid data
        if sft_data is None:
            has_data = False
        else:
            has_data = True

        # 3. Global synchronization: count how many processes have data
        has_data_tensor = torch.tensor([1 if has_data else 0], dtype=torch.long, device=get_device_id())
        torch.distributed.all_reduce(has_data_tensor, op=torch.distributed.ReduceOp.SUM)
        total_workers_with_data = has_data_tensor.item()

        if self.rank == 0:
            print(f"Workers with data: {total_workers_with_data}/{self.world_size}")

        # 4. If globally no data exists, all processes return uniformly
        if total_workers_with_data == 0:
            metrics = {
                "reward/loss": 0.0,
                "reward/grad_norm": 0.0,
                "reward/lr": 0.0
            }
            return DataProto(batch=None, meta_info={"metrics": metrics})

        # 5. 🔥 Key step: handle data broadcasting and padding
        #    - If some processes lack data, perform broadcasting
        #    - All processes use pad_dataproto_to_divisor to ensure batch size is divisible by world_size
        sft_data = self._get_data_from_other_rank(sft_data)

        # 6. Verify all processes now have valid data and batch size divisible by world_size
        if sft_data is None or len(sft_data.batch.get("input_ids", [])) == 0:
            raise RuntimeError(f"Rank {self.rank}: sft_data is invalid after processing")

        # 🔥 Verify all processes have consistent batch sizes divisible by world_size
        final_batch_size = len(sft_data.batch['input_ids'])
        batch_size_tensor = torch.tensor([final_batch_size], dtype=torch.long, device=get_device_id())
        all_final_batch_sizes = [torch.zeros(1, dtype=torch.long, device=get_device_id()) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_final_batch_sizes, batch_size_tensor)

        if self.rank == 0:
            batch_sizes_str = ", ".join([f"rank{i}:{bs.item()}" for i, bs in enumerate(all_final_batch_sizes)])
            print(f"Final batch sizes after padding: {batch_sizes_str}")

            # Check if all batch sizes are identical and divisible by world_size
            all_same = all(bs.item() == final_batch_size for bs in all_final_batch_sizes)
            if not all_same:
                print(f"WARNING: Batch sizes are not consistent across ranks!")

            print(f"Rank {self.rank}: Final data length = {final_batch_size}")

        # 7. Load model (all processes)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.reward_optimizer, device_id=get_device_id())

        # 8. Data preprocessing (all processes)
        if self._do_switch_chat_template:
            sft_data = self._switch_chat_template(sft_data)

        sft_data = sft_data.to(get_device_id())

        # 9. Zero gradients (all processes)
        self.reward_optimizer.zero_grad()

        # 10. Forward + backward pass (all processes execute the full procedure)
        with self.ulysses_sharding_manager:
            sft_data = self.ulysses_sharding_manager.preprocess_data(data=sft_data)

            with Timer(name="update_reward", logger=None) as timer:
                metrics = self.reward.update_reward(data=sft_data)

            # 11. Learning rate scheduling (all processes)
            self.reward_lr_scheduler.step()
            lr = self.reward_lr_scheduler.get_last_lr()[0]
            metrics["reward/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # 12. Offload model (all processes)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        """
        Compute RM scores – adapted for SequenceClassification models

        Important: SequenceClassification requires the full sequence (prompt + response) for scoring.
        Cannot use prompts alone; must use complete input_ids.
        """
        # Process data
        data = self.collaborate.preprocess_reward_data(data)

        # Support all hardwares
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        # ===== Critical modification: use full sequences instead of prompts only =====
        rm_input_ids = data.batch["input_ids"]  # Full sequences
        seq_len = rm_input_ids.shape[1]

        # Use attention_mask and position_ids from the full sequence
        rm_attention_mask = data.batch["attention_mask"][:, :seq_len]
        rm_position_ids = data.batch["position_ids"][:, :seq_len]

        rm_inputs = {
            "input_ids": rm_input_ids,
            "attention_mask": rm_attention_mask,
            "position_ids": rm_position_ids,
        }
        rm_data = DataProto.from_dict(rm_inputs)

        rm_data.meta_info["micro_batch_size"] = self.config.forward_micro_batch_size_per_gpu
        rm_data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        rm_data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(get_device_id())

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            scores = self.reward.compute_rewards(data=rm_data)
            output = DataProto.from_dict(tensors={"rm_scores": scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step,
                                                max_ckpt_to_keep=max_ckpt_to_keep)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path,
                                                del_local_after_load=del_local_after_load)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            if isinstance(data.non_tensor_batch["raw_prompt"][i], str):
                chat: str = data.non_tensor_batch["raw_prompt"][i]
            else:
                raise ValueError("raw_prompt must be a string here")

            prompt_with_chat_template = chat  # This does not require a chat template

            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(
                prompt_with_chat_template, max_length=max_length,
                return_tensors="pt", add_special_tokens=False,
                truncation=True, padding_side="left", padding="max_length"
            )
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=True,
                truncation="left",
            )

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids, "labels": data.batch["labels"]}

        return DataProto.from_dict(rm_inputs)


