######################## 参数设定 ########################
###### 用户设定
# export WANDB_API_KEY=
# export WANDB_PROJECT=SFT
export WANDB_DISABLED=True
EXPERIMENT_TOPIC='binary_classification'  # 当前系列实验主题是什么，可以和其他主题进行区分
EXPERIMENT_NOTES='lora_2B' # 当前实验的备注，可以写一些便于区分的信息

#### LLaMA Factory自带参数
# Model
MODEL_TYPE='qwen2-vl-2b'
MODEL_NAME_OR_PATH="$Qwen2-VL-2B-Instruct"

# Method
LORA_RANK=64
LORA_ALPHA=16
LORA_DROPOUT=0.05
DEEPSPEED=src/sft/configs/deepspeed/ds_z3_config.json

# Dataset
DATASET='risk'  # dataset名字，用于在dataset_info.json登记
DATASET_FILE_PATH='$jsonl_path'
CUTOFF_LEN=802816
MAX_SAMPLES=2000000

# Train
PER_DEVICE_TRAIN_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1.0e-4
NUM_TRAIN_EPOCHS=20
LR_SCHEDULER_TYPE=cosine
WARMUP_RATIO=0.1
BF16=true
DDP_TIMEOUT=180000000
FLASH_ATTN=fa2
ENABLE_LIGER_KERNEL=true

# Output
OUTPUT_ROOT='checkpoint'
OUTPUT_DIR=${OUTPUT_ROOT}/2B_Binary-Guard/${EXPERIMENT_TOPIC}/${EXPERIMENT_TOPIC}_${MODEL_TYPE}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}_${NUM_TRAIN_EPOCHS}epoch_${EXPERIMENT_NOTES}
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=20


######################## 生成本次训练使用的yaml文件 ########################
python src/sft/utils/yaml_generator.py \
          --model_name_or_path=${MODEL_NAME_OR_PATH} \
          --lora_rank=${LORA_RANK} \
          --lora_alpha=${LORA_ALPHA} \
          --lora_dropout=${LORA_DROPOUT} \
          --deepspeed=${DEEPSPEED} \
          --dataset=${DATASET} \
          --cutoff_len=${CUTOFF_LEN} \
          --max_samples=${MAX_SAMPLES} \
          --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
          --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
          --learning_rate=${LEARNING_RATE} \
          --num_train_epochs=${NUM_TRAIN_EPOCHS} \
          --lr_scheduler_type=${LR_SCHEDULER_TYPE} \
          --warmup_ratio=${WARMUP_RATIO} \
          --bf16=${BF16} \
          --ddp_timeout=${DDP_TIMEOUT} \
          --flash_attn=${FLASH_ATTN} \
          --enable_liger_kernel=${ENABLE_LIGER_KERNEL} \
          --output_dir=${OUTPUT_DIR} \
          --save_steps=${SAVE_STEPS} \
          --save_total_limit=${SAVE_TOTAL_LIMIT} \
          

######################## 根据当前数据集在dataset_info.json中登记本次数据集 ########################
python src/sft/utils/dataset_info_generator.py \
          --dataset=${DATASET} \
          --dataset_file_path=${DATASET_FILE_PATH}


######################## 完成所有准备，启动训练 ########################
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 llamafactory-cli train src/sft/train/qwen2vl_lora_sft_trainUsed.yaml