export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
TRAIN_STEP=500
MODEL_DIR='$Qwen2-VL-2B-Instruct'
LORA_ADAPTOR_DIR='./checkpoint/2B_Binary-Guard/checkpoint-'${TRAIN_STEP}''
EXPERIMENT_NOTES='binary_classification'
DATA_ROOT_DIR='$data_path'
INFER_FILE_NAME='merged'
INFER_FILE_TYPE='.jsonl'

MAX_NEW_TOKENS=1024
PER_IMG_MAX_TOKENS=1024
PER_IMG_MIN_TOKENS=256
DO_SAMPLE=False

INFER_FILE_PATH=${DATA_ROOT_DIR}/${INFER_FILE_NAME}${INFER_FILE_TYPE}
OUTPUT_FILE_PATH=./reports/${EXPERIMENT_NOTES}/${INFER_FILE_NAME}'_'${TRAIN_STEP}'Steps.json'
REPORT_DIR=./reports/${EXPERIMENT_NOTES}/${INFER_FILE_NAME}'_'${TRAIN_STEP}'Steps'

if [ ! -d "${REPORT_DIR}" ]; then
  mkdir -p "${REPORT_DIR}"
  echo "Directory created: ${REPORT_DIR}"
else
  echo "Directory already exists: ${REPORT_DIR}"
fi

torchrun --nproc_per_node=8 inference_ddp.py \
            --model_dir=${MODEL_DIR} \
            --lora_adaptor_dir=${LORA_ADAPTOR_DIR} \
            --max_new_token=${MAX_NEW_TOKENS} \
            --per_img_max_tokens=${PER_IMG_MAX_TOKENS} \
            --per_img_min_tokens=${PER_IMG_MIN_TOKENS} \
            --infer_batchsize 16 \
            --do_sample=${DO_SAMPLE} \
            --infer_file_path=${INFER_FILE_PATH} \
            --output_file_path=${OUTPUT_FILE_PATH}
