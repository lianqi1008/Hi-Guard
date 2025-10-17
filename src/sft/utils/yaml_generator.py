import yaml
from args import parse_global_args

args = parse_global_args()

if __name__ == '__main__':
    # 打开并读取YAML文件
    with open('src/sft/train/qwen2vl_lora_sft_base.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        for arg, value in vars(args).items():
            if config.get(arg, None) is not None:
                config[arg] = value

        # Save dictionary to YAML
        with open('src/sft/train/qwen2vl_lora_sft_trainUsed.yaml', 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
