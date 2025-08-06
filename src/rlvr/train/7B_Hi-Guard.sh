export DEBUG_MODE="true"
export LOG_PATH="./debug/7B_Hi-Guard.txt"

export DATA_FILE_PATH=$json_path
export CKPT_PATH=$Qwen2-VL-7B-Instruct
export SAVE_PATH=./share_models/7B_Hi-Guard/

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/rlvr/src/open_r1/Hi-Guard.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --deepspeed src/rlvr/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --save_steps 400 \
    --save_only_model true \
    --num_generations 4 \
    --dataset_name classfication  \
    --dataset_file_path ${DATA_FILE_PATH} \
    --report_to tensorboard \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --max_pixels 802816