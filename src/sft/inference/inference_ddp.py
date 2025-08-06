import sys
import os
sys.path.append(os.getcwd())

import torch
import argparse
import json
import time
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from pathlib import Path
import jsonlines
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from scripts.vision_process import process_vision_info
from modelscope import snapshot_download

def initialize_distributed():
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

def setup_argparser():
    parser = argparse.ArgumentParser(description="A simple program to demonstrate command line flags.")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--lora_adaptor_dir', type=str, default=None)
    parser.add_argument('--max_new_token', type=int, default=10)
    parser.add_argument('--per_img_max_tokens', type=int, default=1280)
    parser.add_argument('--per_img_min_tokens', type=int, default=256)
    parser.add_argument('--infer_batchsize', type=int, default=1)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--output_logits', type=bool, default=True)
    parser.add_argument('--infer_file_path', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=True)
    return parser.parse_args()

class MessageDataset(Dataset):
    def __init__(self, messages):
        self.messages = messages
        self.convert_llamafactory_to_hf()

    def convert_llamafactory_to_hf(self):
        for message in tqdm(self.messages, desc="Convert DataSet From LlamaFactoryType to HuggingfaceType"):
            assert message['conversations'][0]['from'] == "human"
            prompt = message['conversations'][0]['value']
            if '<image>' in prompt: 
                images = message['images'] + ['']
                prompt_lst = prompt.split('<image>')
                if len(images) != len(prompt_lst):
                    print(message['images'])
                    print(len(images), images)
                    print(len(prompt_lst), prompt_lst)
                assert len(images) == len(prompt_lst)
                content = []
                for image, prompt in zip(images, prompt_lst):
                    if len(prompt) > 0:
                        content.append({"type": "text", "text": prompt})
                    if len(image) > 0:
                        content.append({"type": "image", "image": image})
            else:
                content = []
                content.append({"type": "text", "text": prompt})

            message['model_input'] = [
                                        {
                                            "role": "system",
                                            "content": message['system'],
                                        },
                                        {
                                            "role": "user",
                                            "content": content
                                        }
                                    ]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]

def collate_fn(batch):
    return {
        'model_input': [b['model_input'] for b in batch],
        'raw': batch
    }

def load_messages(file_path):
    if Path(file_path).suffix == '.json':
        return json.load(open(file_path, 'r'))
    elif Path(file_path).suffix == '.jsonl':
        with jsonlines.open(file_path) as reader:
            return [obj for obj in reader]
    else:
        raise ValueError('Invalid file format. Only JSON and JSONL are supported.')

def setup_model(args):
    if args.lora_adaptor_dir:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, attn_implementation="flash_attention_2",torch_dtype=torch.float16)
        return PeftModel.from_pretrained(base_model, args.lora_adaptor_dir)
    return Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

def create_processor(args, model_dir):
    min_pixels = args.per_img_min_tokens * 14 * 14
    max_pixels = args.per_img_max_tokens * 14 * 14
    return AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

def main():
    args = setup_argparser()
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank != -1:
        initialize_distributed()

    # 初始化模型和处理器
    model = setup_model(args)
    model.to('cuda')
    processor = create_processor(args, args.model_dir)

    # 加载输入数据
    messages = load_messages(args.infer_file_path)
    dataset = MessageDataset(messages)
    sampler = torch.utils.data.DistributedSampler(dataset) if local_rank != -1 else None
    dataloader = DataLoader(dataset,
                            batch_size=args.infer_batchsize,
                            sampler=sampler,
                            collate_fn=collate_fn,
                            shuffle=False,
                            num_workers=multiprocessing.cpu_count(),
                            prefetch_factor=args.infer_batchsize * 10)

    local_outputs = []
    infer_step = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc='预测中')):
            start_time = time.time()
            
            input_batch = batch['model_input']

            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in input_batch]
            image_inputs, video_inputs = process_vision_info(input_batch)
            inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to('cuda')

            # 模型推理
            output = model.generate(**inputs, max_new_tokens=args.max_new_token, do_sample=args.do_sample, return_dict_in_generate=True, output_logits=args.output_logits)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], output.sequences)]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for b, out_text in zip(batch['raw'], output_texts):
                result = {'response_text': out_text}
                b['llm_response'] = result
                local_outputs.append(b)
            
    all_outputs = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(all_outputs, local_outputs)
    
    if torch.distributed.get_rank() == 0:
        with open(args.output_file_path, 'w') as f:
            combined_outputs = [item for sublist in all_outputs for item in sublist]
        
            results = []
            seen = set()
            for d in combined_outputs:
                image = d["images"][0]
                if image not in seen:
                    results.append(d)
                    seen.add(image)
                else:
                    print(image)
           
            json.dump(results, f, ensure_ascii=False, indent=4)
            print('Final Result Has Been Writted in File: {}'.format(args.output_file_path))

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
