import io
import os
import re
import json
import sys
import math
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer,
                          AutoProcessor, StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
import torch.nn.functional as F
torch.manual_seed(1234)
import time
import pandas as pd
from collections import defaultdict
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
from statistics import mean
from datetime import timedelta

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# add absolute path
# sys.path.append('absolute_path')
from vision_process import process_vision_info

generation_config = GenerationConfig(
    max_new_tokens=1024,
    repetition_penalty=1.15,
)

""""
==========================================变量定义=========================================
"""
# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

# 在变量定义部分添加数据集映射
DATASET_LABEL_MAPPING = {
    'dataset1': {
        'label1': 0,
        'label2': 1,
        'label3': 2,
    },
    'dataset2': {
        'label1': 0,
        'label2': 1,
        'label3': 2,
        'label4': 3,
    }
}

SYSTEM_PROMPT_dataset1 = """你是一名内容风险审核专家，任务是对给定内容进行风险分类，并输出详细推理理由作为证据。

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

SYSTEM_PROMPT_dataset2= """你是一名内容风险审核专家，任务是对给定内容进行风险分类，并输出详细推理理由作为证据。

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

SYSTEM_PROMPT_MAPPING = {'dataset1': SYSTEM_PROMPT_dataset1, 'dataset2': SYSTEM_PROMPT_dataset2}

""""
==========================================推理=========================================
"""
def save_predictions(predictions_data, filepath):
    """保存预测结果到文件"""
    # 保存为JSON格式
    json_filepath = filepath.replace('.pkl', '.json')
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    
    # 同时保存为pickle格式（更快的加载速度）
    with open(filepath, 'wb') as f:
        pickle.dump(predictions_data, f)
    
    print(f"预测结果已保存到: {json_filepath} 和 {filepath}")

def load_predictions(filepath):
    """从文件加载预测结果"""
    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        print("load error")

""""
==========================================推理=========================================
"""
def extract_prediction_from_response(response):
    """从模型响应中提取预测结果"""
    try:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer_content = match.group(1).strip()
            return answer_content.replace(' ', '').replace('_', '').split('-')[-1], True
        else:
            return None, False
    except Exception as e:
        return None, False

def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://", timeout=timedelta(minutes=30))

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_parallel(rank, world_size, json_file_path, model_base, model_path, report_dir, SYSTEM_PROMPT):
    """并行执行函数"""

    setup_distributed(rank, world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_base)
    
    with open(json_file_path, 'r') as f:
        val_set = json.load(f)
    
    total_samples = len(val_set)
    samples_per_gpu = total_samples // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else total_samples
    
    local_val_set = val_set[start_idx:end_idx]
    if rank == 0: 
        print(f"Total samples: {total_samples}. World size: {world_size}")
    print(f"GPU {rank} processing samples {start_idx}-{end_idx-1} ({len(local_val_set)} samples)")
    
    format_error_file = os.path.join(report_dir, f"format_error_cases_rank_{rank}.jsonl")
    answer_error_file = os.path.join(report_dir, f"answer_error_cases_rank_{rank}.jsonl")
    answer_right_file = os.path.join(report_dir, f"answer_right_cases_rank_{rank}.jsonl")
    
    open(format_error_file, 'w').close()
    open(answer_error_file, 'w').close()
    open(answer_right_file, 'w').close()
    
    predictions_data = []
    format_correct_count = 0
    all_correct_count = 0

    report_interval = 10

    # MODIFIED: Initialize buffer for rank 0
    global_stats_log_buffer = [] if rank == 0 else None

    for idx, row in enumerate(tqdm(local_val_set, desc=f"GPU {rank} Validating", unit="row", disable=(rank!=0))):
        global_idx = start_idx + idx
        
        image_cate_match = re.search(r'<answer>(.*?)</answer>', row["solution"])
        if not image_cate_match:
            logger.warning(f"GPU {rank}: Could not extract ground truth from solution: {row['solution']} for sample {global_idx}. Skipping.")
            continue
        image_cate = image_cate_match.group(1).strip()
        image_cate = image_cate.split('-')[-1]

        messages = [
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
                    {"type": "image", "image": row["images"][0]},
                    {"type": "text", "text": row["text"]},
                ]
            },
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(**inputs, generation_config=generation_config, use_cache=True)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        prediction, format_valid = extract_prediction_from_response(response)
        image_cate_clean = str(image_cate).replace(' ', '').replace('_', '')
        
        text_content = row["text"]
        if text_content:
            text_content = text_content.replace('\n', ' ').replace('\r', ' ')
            text_content = ' '.join(text_content.split())
        else:
            text_content = ''

        prediction_record = {
            "sample_idx": global_idx, "ground_truth": image_cate_clean, "prediction": prediction,
            "raw_response": response, "format_valid": format_valid, "image_path": row["images"][0],
            "text": row["text"], 
            "correct": prediction == image_cate_clean if prediction is not None and format_valid else False
        }
        predictions_data.append(prediction_record)

        if format_valid:
            case_info_for_log = {
                "idx": global_idx, 
                "ground_truth": image_cate_clean, 
                "prediction": prediction,
                "response": response,
                "image_path": os.path.basename(row["images"][0]), 
                "text": text_content,
            }
            format_correct_count += 1
            if prediction == image_cate_clean:
                all_correct_count += 1
                with open(answer_right_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(case_info_for_log, ensure_ascii=False) + "\n")
            else:
                with open(answer_error_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(case_info_for_log, ensure_ascii=False) + "\n")
        else:
            case_info_for_log = {
                "idx": global_idx, "ground_truth": image_cate_clean, "response": response,
                "image_path": os.path.basename(row["images"][0]), "text": text_content,
            }
            with open(format_error_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(case_info_for_log, ensure_ascii=False) + "\n")

        if (idx + 1) % report_interval == 0 or idx == len(local_val_set) - 1:
            # 所有GPU都需要参与通信
            local_processed = idx + 1
            local_format_correct = format_correct_count
            local_all_correct = all_correct_count
            
            # 创建张量用于通信
            stats_tensor = torch.tensor([local_processed, local_format_correct, local_all_correct], 
                                    dtype=torch.float32, device=device)
            
            # 所有GPU都参与收集统计信息
            gathered_stats = [torch.zeros_like(stats_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_stats, stats_tensor)
            
            # 只有rank 0负责计算和输出
            if rank == 1:
                # 计算全局统计
                total_processed = sum([stats[0].item() for stats in gathered_stats])
                total_format_correct = sum([stats[1].item() for stats in gathered_stats])
                total_all_correct = sum([stats[2].item() for stats in gathered_stats])
                
                # 计算比率
                global_format_rate = total_format_correct / total_processed if total_processed > 0 else 0
                global_accuracy = total_all_correct / total_processed if total_processed > 0 else 0
                global_progress = total_processed / total_samples
                
                print(f"\n{GREEN}=== 全局实时统计 ({int(total_processed)}/{total_samples}) ==={RESET}")
                print(f"{GREEN}格式正确率: {global_format_rate:.4f} ({int(total_format_correct)}/{int(total_processed)}){RESET}")
                print(f"{GREEN}整体准确率: {global_accuracy:.4f} ({int(total_all_correct)}/{int(total_processed)}){RESET}")
                print(f"{GREEN}全局进度: {global_progress:.2%}{RESET}")
                print("-" * 60)

    predictions_file_rank = os.path.join(report_dir, f"predictions_rank_{rank}.pkl")
    save_predictions(predictions_data, predictions_file_rank)
    
    local_format_correct_rate = format_correct_count / len(local_val_set) if len(local_val_set) > 0 else 0
    local_accuracy_rate = all_correct_count / len(local_val_set) if len(local_val_set) > 0 else 0
    
    cleanup_distributed()
    
    return {
        'rank': rank, 'format_correct': local_format_correct_rate, 'accuracy': local_accuracy_rate,
        'total_samples': len(local_val_set), 'predictions_file': predictions_file_rank
    }

def merge_prediction_files(report_dir, world_size):
    all_predictions = []
    for rank in range(world_size):
        rank_file = os.path.join(report_dir, f"predictions_rank_{rank}.pkl")
        if os.path.exists(rank_file):
            try:
                rank_predictions = load_predictions(rank_file)
                all_predictions.extend(rank_predictions)
                os.remove(rank_file)
                json_rank_file = rank_file.replace('.pkl', '.json')
                if os.path.exists(json_rank_file):
                    os.remove(json_rank_file)
            except Exception as e:
                logger.error(f"Error loading or removing prediction file {rank_file}: {e}")
    
    all_predictions.sort(key=lambda x: x['sample_idx'])
    merged_file = os.path.join(report_dir, "predictions.pkl")
    save_predictions(all_predictions, merged_file)
    return all_predictions

def merge_files(report_dir, world_size):
    file_types = ['format_error_cases', 'answer_error_cases', 'answer_right_cases']
    for file_type in file_types:
        merged_file_path = os.path.join(report_dir, f"{file_type}.jsonl")
        with open(merged_file_path, 'w', encoding='utf-8') as merged_f:
            for rank in range(world_size):
                rank_file_path = os.path.join(report_dir, f"{file_type}_rank_{rank}.jsonl")
                if os.path.exists(rank_file_path):
                    try:
                        with open(rank_file_path, 'r', encoding='utf-8') as f_rank:
                            merged_f.write(f_rank.read())
                        os.remove(rank_file_path)
                    except Exception as e:
                         logger.error(f"Error merging or removing file {rank_file_path}: {e}")

def save_prediction_summary(predictions_data, report_dir):
    predictions_summary = []
    for record in predictions_data:
        text_orig = record['text'] # Use original text from record
        text_display = ''
        if text_orig:
            text_display = str(text_orig)[:100] + '...' if len(str(text_orig)) > 100 else str(text_orig)
            text_display = text_display.replace('\n', ' ').replace('\r', ' ')
        
        predictions_summary.append({
            'sample_idx': record['sample_idx'],
            'ground_truth': record['ground_truth'],
            'prediction': record['prediction'] if record['prediction'] is not None else 'NULL',
            'valid_prediction': True if record['prediction']==record['ground_truth'] else False,
            'correct': record['correct'],
            'image_path': os.path.basename(record['image_path']),
            'text': text_display # Use truncated and cleaned text for summary
        })
    
    pd.DataFrame(predictions_summary).to_csv(os.path.join(report_dir, 'predictions_summary.csv'), index=False, encoding='utf-8-sig')


def main():
    model_base, model_path, json_file_path, report_dir, dataset = sys.argv[1:6]
    predictions_file = None

    print(f'model_base: {model_base}\nmodel_path: {model_path}\njson_file_path: {json_file_path}\nreport_dir:{report_dir}\n')
        
    if not os.path.exists(report_dir):
        os.makedirs(report_dir, exist_ok=True)
        print(f"Created report directory: {report_dir}")
    
    # 根据dataset获取对应的标签字典
    label_dict = DATASET_LABEL_MAPPING[dataset]
    print(f"使用数据集: {dataset}")
    print(f"标签数量: {len(label_dict)}\n")

    SYSTEM_PROMPT = SYSTEM_PROMPT_MAPPING[dataset]
    
    n_gpus = torch.cuda.device_count()
    print(f"检测到 {n_gpus} 个GPU")
    
    # MODIFIED: Clear real_time_global_stats.txt at the beginning of a new multi-GPU run
    if n_gpus > 1 : 
        stats_log_file_path = os.path.join(report_dir, 'real_time_global_stats.txt')
        if os.path.exists(stats_log_file_path):
            open(stats_log_file_path, 'w').close() 
            print(f"Cleared existing real_time_global_stats.txt for new multi-GPU run.")


    if n_gpus > 1:
        logger.info(f'Started generation in multi-GPU mode with {n_gpus} GPUs')
        mp.set_start_method('spawn', force=True)
        
        # 开始计时
        start_time = time.time()

        with mp.Pool(processes=n_gpus) as pool:
            results = pool.starmap(
                run_parallel,
                [(rank, n_gpus, json_file_path, model_base, model_path, report_dir, SYSTEM_PROMPT) for rank in range(n_gpus)]
            )
        
        # 结束计时
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        predictions_data = merge_prediction_files(report_dir, n_gpus)
        # 保存耗时到文件
        with open(os.path.join(report_dir, "elapsed_time.txt"), "w") as f:
            f.write(f"Elapsed time: {elapsed_time} seconds")
        merge_files(report_dir, n_gpus)
    else:
        logger.error("No GPUs detected. Exiting.")
        sys.exit(1)
    
    print(f"\n运行完成！预测结果已保存")

if __name__ == "__main__":
    main()