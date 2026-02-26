import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from functools import partial
import torch
from torch.nn import MSELoss
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


# ============================================================================================================
# QWEN-7B
exp_name = "Cold-Start-MetaRM-FSDP-Qwen-7B"
model_path = "/path/to/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
data_path = "/path/to/helpsteer3_raw_full_train_rl_modified_human_critique_rl_A_base_model_rollout_4_sft_format.jsonl"
transformer_layer_cls_to_wrap = ["Qwen2DecoderLayer"]

# LLAMA-8B
# exp_name = "Cold-Start-MetaRM-FSDP-Llama-8B"
# model_path = "/path/to/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# data_path = "/path/to/helpsteer3_raw_full_train_rl_modified_human_critique_rl_A_base_model_rollout_4_llama_8b_sft_format.jsonl"
# transformer_layer_cls_to_wrap = ["LlamaDecoderLayer"]

# QWEN-32B
# exp_name = "Cold-Start-MetaRM-FSDP-Qwen-32B"
# model_path = "/path/to/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# data_path = "/path/to/helpsteer3_raw_full_train_rl_modified_human_critique_rl_A_base_model_rollout_4_qwen_32b_sft_format.jsonl"
# transformer_layer_cls_to_wrap = ["Qwen2DecoderLayer"]

final_model_save_path = f"/path/to/checkpoints/{exp_name}"
# ============================================================================================================


# Environment variables for logging (replace with your own keys if needed)
os.environ["SWANLAB_API_KEY"] = "your_swanlab_api_key_here"
os.environ["SWANLAB_PROJECT"] = "your_project_name_here"
os.environ["SWANLAB_EXPERIMENT_NAME"] = exp_name


def read_jsonl_safe(file_path):
    """Safely read a JSONL file, skipping malformed lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {i}: {e}")
                continue


@dataclass
class RegressionDataCollator:
    """Custom data collator for regression tasks that handles labels correctly."""
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract labels before padding
        labels = [feature.pop("labels") for feature in features]

        # Pad inputs using the tokenizer
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        # Add labels back as float tensor
        batch["labels"] = torch.tensor(labels, dtype=torch.float)

        return batch


class RegressionTrainer(Trainer):
    """Custom Trainer for regression tasks."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        loss_fct = MSELoss()
        loss = loss_fct(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics: MSE, RMSE, MAE."""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()

        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }


def apply_template(data_path):
    """Apply prompt template and filter data."""
    data = read_jsonl_safe(data_path)
    filtered_data = []

    for item_dict in data:
        conv_his = item_dict.get("conv_his", "")
        critics = item_dict.get("critics", "")

        prompt = (
            f"<conv_his>{conv_his}</conv_his>\n\n"
            f"<critics>{critics}</critics>"
        )
        item_dict["prompt"] = prompt
        filtered_data.append(item_dict)

    print(f"Loaded {len(filtered_data)} samples after filtering")
    return filtered_data


def process_function(examples, tokenizer):
    """Tokenize prompts and prepare labels."""
    tokenized = tokenizer(
        examples["prompt"],
        max_length=10240,
        truncation=True,
        padding=False
    )

    tokenized["labels"] = [float(label) for label in examples["score"]]
    return tokenized


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side to left (common for decoder-only models)
    tokenizer.padding_side = "left"

    print(f"Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")

    # Load and preprocess data
    raw_data = apply_template(data_path)
    dataset = Dataset.from_list(raw_data)

    # Split into train and validation sets
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Tokenize datasets
    tokenized_train = train_dataset.map(
        partial(process_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = eval_dataset.map(
        partial(process_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Load model for sequence classification (regression with 1 output)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Sync pad_token_id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model pad_token_id: {model.config.pad_token_id}")

    # Training arguments with FSDP enabled
    train_args = TrainingArguments(
        output_dir=final_model_save_path,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,

        # Optimizer settings
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.01,

        # Training loop
        num_train_epochs=3,
        bf16=True,

        # FSDP configuration
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": False,
            "use_orig_params": True,
            "transformer_layer_cls_to_wrap": transformer_layer_cls_to_wrap,
        },

        # Logging & saving
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=40,

        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Reporting
        report_to=["swanlab"],

        # Misc
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Initialize custom trainer
    trainer = RegressionTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=RegressionDataCollator(tokenizer=tokenizer),
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    print(f"Model saved to {final_model_save_path}")


if __name__ == "__main__":
    main()
