"""
This scrip support is adapted from Tora project to annotate the mathematical reasoning data.
"""
from datasets import load_dataset, Dataset, DatasetDict

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datasets import load_dataset
import requests
from eval.evaluate import evaluate
from eval.evaluate import get_batch_scores
from tqdm import tqdm
from utils.data_loader import load_data
from utils.parser import *
from utils.python_executor import PythonExecutor
from utils.utils import construct_prompt, load_jsonl, save_jsonl, set_seed


import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="/home/wx13/dpo_test/math_eval/iter_dpo_numina_rule_reward.jsonl",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="/home/wx13/dpo_test/math_eval/iter_dpo_numina_rule_xxx.jsonl",
        metadata={"help": "the location of the output file"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
ds = load_dataset('json', data_files=script_args.dataset_name_or_path, split="train")

remain_codes = []
remain_gts = []
all_samples = []

for sample in ds:
    remain_codes.append(sample['responses'])
    remain_gts.append(sample['gt'])    
    all_samples.append(sample)

print(remain_codes[0])

all_rm_scores = get_batch_scores(remain_codes, remain_gts)

all_data = []

for i, sample in enumerate(all_samples):
    sample.update({"rewards": all_rm_scores[i]})
    all_data.append(sample)

'''
import numpy as np
print(np.mean([sam['rewards'] for sam in all_data]))
keys = all_data[0].keys()  

dict_data = {key: [d[key] for d in all_data] for key in keys}
output_dir = script_args.output_dir

dataset = Dataset.from_dict(dict_data)
DatasetDict({'train': dataset}).push_to_hub(output_dir)
'''



with open(script_args.output_dir, "w", encoding="utf8") as f:
    for i in range(len(all_data)):
        json.dump(all_data[i], f, ensure_ascii=False)
        f.write('\n')
