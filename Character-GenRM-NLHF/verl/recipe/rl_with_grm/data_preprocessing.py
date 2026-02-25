import copy
import os
import pandas as pd
from datasets import load_dataset

dataset = load_dataset("Tongyi-ConvAI/RM-NLHF")

output_train_dir = "/path/to/train.parquet"
output_test_dir = "/path/to/test.parquet"
os.makedirs(os.path.dirname(output_train_dir), exist_ok=True)

new_data = []

for split_name in ["train"]:
    for obj in dataset[split_name]:
        extra_info = obj.get("extra_info", {})
        if isinstance(extra_info, str):
            import json
            extra_info = json.loads(extra_info)

        if extra_info.get("dataset_type") != "DataWithHumanCritique":
            continue

        new_obj = {
            "prompt": extra_info.get("context", ""),
            "data_source": "helpsteer3",
            "ability": "other",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": copy.deepcopy(obj)
        }
        new_data.append(new_obj)

print(f"dataset length: {len(new_data)}")

pd.DataFrame(new_data).to_parquet(output_train_dir, index=False)
pd.DataFrame(new_data[:128]).to_parquet(output_test_dir, index=False)

print(f"Train saved to: {output_train_dir}")
print(f"Test  saved to: {output_test_dir}")
