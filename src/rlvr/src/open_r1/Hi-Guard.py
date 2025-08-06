# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import sys
# parent directory path of trainer
# sys.path.insert(0, '/mnt/tidal-alsh01/usr/lianqi4/Hi-Guard/src/rlvr/src')
# print(os.getcwd())
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import json
from collections import Counter
from math_verify import parse, verify
import inspect
import wandb
import functools
import logging

# Configure logging at the top of your file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import Qwen2VLForConditionalGeneration

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import numpy as np
import re

# Replace with your API key
# wandb.login(key="")

def clean_text(text):
    return text.strip().replace(' ', '').replace('_', '').lower() if text else ""

def label_distance_matrix(label_to_parent):
    labels = list(label_to_parent.keys())
    n = len(labels)
    dist = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li == lj:
                dist[i, j] = 0
            elif label_to_parent[li] == label_to_parent[lj]:
                dist[i, j] = 1
            else:
                dist[i, j] = 2
    return labels, dist

def soft_margin_penalty(pred_label, true_label, label2idx, dist_matrix):
    pred_idx = label2idx.get(pred_label, -1)
    true_idx = label2idx.get(true_label, -1)
    if pred_idx == -1 or true_idx == -1:
        return 0.0
    distance = dist_matrix[true_idx, pred_idx]
    if pred_label == true_label:
        return 0.0
    elif distance == 1:
        return 1.0
    elif distance == 2:
        return 0.5
    return 0.0


CATEGORY_PATHS = [
    "level1-level2-level3-level4"
]

# Four-level label structure
level_names = [1, 2, 3, 4]


# Construct the set of all labels for each layer
level_labels = {name: set() for name in level_names}
for path in CATEGORY_PATHS:
    splits = path.split("-")
    for i, name in enumerate(level_names):
        level_labels[name].add(splits[i])

# Build a label -> parent mapping for each layer
label_to_parent_per_level = {name: {} for name in level_names}
for path in CATEGORY_PATHS:
    splits = path.split("-")
    for i, name in enumerate(level_names):
        if i == 0:
            label_to_parent_per_level[name][splits[i]] = None  # Level 1 has no parent class
        else:
            label_to_parent_per_level[name][splits[i]] = splits[i-1]

# Construct a label -> idx mapping for each layer
label2idx_per_level = {
    name: {label: idx for idx, label in enumerate(sorted(level_labels[name]))}
    for name in level_names
}

# Distance matrix function (reuse above)
def label_distance_matrix(label_to_parent):
    labels = list(label_to_parent.keys())
    n = len(labels)
    dist = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li == lj:
                dist[i, j] = 0
            elif label_to_parent[li] == label_to_parent[lj]:
                dist[i, j] = 1
            else:
                dist[i, j] = 2
    return labels, dist

dist_matrix_per_level = {}
for name in level_names:
    labels, dist = label_distance_matrix(label_to_parent_per_level[name])
    dist_matrix_per_level[name] = dist

class RewardCalculator:
    def __init__(self, label_to_parent_per_level, label2idx_per_level, dist_matrix_per_level):
        # Initialize multi-level information
        self.label_to_parent_per_level = label_to_parent_per_level
        self.label2idx_per_level = label2idx_per_level
        self.dist_matrix_per_level = dist_matrix_per_level

    def accuracy_reward_optimized(self, completions, solution, **kwargs):
        """
        completions: List of model outputs, one per sample
        solution: List of ground truth, one per sample
        **kwargs: 必须包含每一层标签的预测与GT字典、think内容、层级名
        """
        rewards = []
        for idx, (completion, sol) in enumerate(zip(completions, solution)):
            try:
                total_reward = 0.0
                content = completion[0]["content"]
                pred_match = re.search(r"<answer>(.*?)</answer>", content)
                sol_match = re.search(r"<answer>(.*?)</answer>", sol)
                if pred_match and sol_match:
                    pred_answer = clean_text(pred_match.group(1))
                    true_answer = clean_text(sol_match.group(1))
                    if true_answer=='无风险' or pred_answer=='无风险':
                        base_acc = 1.0 if pred_answer == true_answer else -0.5
                        total_reward = base_acc
                    else:
                        pred_answers = pred_answer.split('-')
                        true_answers = true_answer.split('-')
                        length = min(len(pred_answers), len(true_answers))
                        for idx, (pred, true) in enumerate(zip(pred_answers[:length], true_answers[:length])):
                            label2idx = self.label2idx_per_level[idx+1]
                            dist_matrix = self.dist_matrix_per_level[idx+1]
                            base_acc = 1.0 if pred == true else 0.0
                            sim_penalty = soft_margin_penalty(pred, true, label2idx, dist_matrix)
                            reward = (base_acc - 0.5*sim_penalty)
                            total_reward += reward
                        total_reward /= length
                rewards.append(total_reward)
            except Exception as e:
                print(f"Reward calculation error: {e}")
                rewards.append(0.0)
        return rewards

    def format_reward_optimized(self, completions, solution, **kwargs):
        rewards = []
        for completion, sol in zip(completions, solution):
            content = completion[0]["content"]
            reward = 0.0
            pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
            try:
                match = re.fullmatch(pattern, content, re.DOTALL)
                if match:
                    reward = 1.0
            except Exception as e:
                print(f"Error in format_reward: {e}")
            rewards.append(reward)
        return rewards

reward_calculator = RewardCalculator(label_to_parent_per_level, label2idx_per_level, dist_matrix_per_level)

# ======= 奖励函数定义 =======
def accuracy_reward(completions, solution, **kwargs):
    """准确性奖励函数"""
    return reward_calculator.accuracy_reward_optimized(completions, solution, **kwargs)

def format_reward(completions, solution, **kwargs):
    """格式奖励函数"""
    return reward_calculator.format_reward_optimized(completions, solution, **kwargs)

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

# ====== 脚本参数定义 ======

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "quality"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    dataset_file_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset file"}
    )

# ===== Dataset构造优化 (只一次) =====

SYSTEM_PROMPT="""你是一名内容风险审核专家，任务是对给定内容进行风险分类，并输出详细推理理由作为证据。

你的决策路径包含四级类别，平台已为每个类别提供权威定义。你应综合每一级别的类别定义，尤其关注与当前标签的**父类定义**和**同父类下的其他兄弟类别定义**的区别点，明确说明理由，防止混淆。

【类别定义】

一级类别：
...

二级类别：
...

三级类别：
...

四级类别：
...

【分类路径选择】

...
无风险

【分类要求】
1. 必须从给定的所有风险路径中选择最符合内容的四级风险路径。如果无风险，则选择“无风险”。
2. 推理（<think>）中要体现：你为何认定该内容属于该分类？与同父类下兄弟类别的**区别**是什么？（如“与XX类别不同，该内容有XX风险点/细节”）
3. 推理过程不少于20字，且需引用类别定义中的风险点、关键要素，但不能简单照抄定义文本。
4. 最终答案只填写具体四级路径或“无风险”。


请严格按照以下格式输出：

<think>推理过程：详细解释你识别该分类的理由，重点突出风险点及与相似类别的区别，至少20字。</think>
<answer>四级风险路径 或 无风险</answer>
"""

def make_conversation_image(example):
    return {
        "prompt": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "以下是需要判断的内容，请给出分类判断和理由："},
                    {"type": "image"},
                    {"type": "text", "text": example["text"]},
                ]
            },
        ]
    }

def main(script_args, training_args, model_args):
    script_args.reward_funcs = ['accuracy','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = load_dataset("json", data_files=script_args.dataset_file_path)
    dataset = dataset.map(make_conversation_image) 

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    # print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
