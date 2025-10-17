import json
from args import parse_global_args

args = parse_global_args()

if __name__ == '__main__':
    data_info = json.load(open('src/sft/configs/data/dataset_info.json', 'r', encoding='utf-8-sig'))
    target_data_dict = data_info['qwen2-vl-contentRisk-base']

    target_data_dict['file_name'] = args.dataset_file_path
    data_info[args.dataset] = target_data_dict

    json.dump(data_info, open('src/sft/configs/data/dataset_info.json', 'w'), ensure_ascii=False, indent=4)
