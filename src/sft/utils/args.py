import argparse


def parse_global_args():
    parser = argparse.ArgumentParser(description="Training configuration for Qwen2-VL-2B model.")

    # Model parameters
    parser.add_argument("--model_name_or_path", type=str, default="models/qwen2-vl-2b", help="Path to pre-trained model.")

    # Finetuning method parameters
    parser.add_argument("--stage", type=str, default="sft", help="Training stage.")
    parser.add_argument("--do_train", type=bool, default=True, help="Flag to perform training.")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Type of finetuning.")
    parser.add_argument("--lora_target", type=str, default="all", help="LoRA target layers.")
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank for LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA.")
    parser.add_argument("--deepspeed", type=str, default="examples/deepspeed/ds_z2_config.json", help="Path to DeepSpeed configuration.")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="", help="Name of the dataset.")
    parser.add_argument("--dataset_file_path", type=str, default="")
    parser.add_argument("--template", type=str, default="qwen2_vl", help="Template for dataset.")
    parser.add_argument("--cutoff_len", type=int, default=3072, help="Cutoff length for input sequences.")
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples.")
    parser.add_argument("--overwrite_cache", type=bool, default=True, help="Flag to overwrite cache.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=16, help="Number of workers for data preprocessing.")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="", help="Output directory.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Frequency of logging.")
    parser.add_argument("--save_steps", type=int, default=500, help="Frequency of saving checkpoints.")
    parser.add_argument("--save_total_limit", type=int, default=100, help="Total number of checkpoints to keep.")
    parser.add_argument("--plot_loss", type=bool, default=True, help="Flag to plot loss.")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True, help="Flag to overwrite output directory.")

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=float, default=5.0, help="Number of training epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Type of learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--bf16", type=bool, default=True, help="Flag to use bf16 training.")
    parser.add_argument("--ddp_timeout", type=int, default=180000000, help="Timeout for distributed data parallel training.")
    parser.add_argument("--flash_attn", type=str, default="fa2", help="Flash attention type.")
    parser.add_argument("--enable_liger_kernel", type=bool, default=True, help="Flag to enable Liger kernel.")

    # Merge Lora
    parser.add_argument("--export_dir", type=str, default="lora export dir", help="lora export dir")
    parser.add_argument("--adapter_name_or_path", type=str, default="lora adaptor path", help="Flag to lora adaptor path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_global_args()
    print(args)
