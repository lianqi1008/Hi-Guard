#!/bin/bash

MODEL_BASE="checkpoint/Qwen2-VL-7B-Instruct"
MODEL_PATH="checkpoint/7B_Hi-Guard/checkpoint-xxx"
JSON_FILE_PATH="$json_path"
REPORT_DIR="reports/$(date +%m%d_%H%M)"


# 设置GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# =============================================================================
# 执行脚本
# =============================================================================

echo "开始评估..."
echo "时间: $(date)"
echo "模型: $MODEL_PATH"
echo "数据: $JSON_FILE_PATH"
echo "输出: $REPORT_DIR"
echo ""

# 创建输出目录
mkdir -p "$REPORT_DIR"

# 运行Python脚本
python src/rlvr/inference/inference.py \
    "$MODEL_BASE" \
    "$MODEL_PATH" \
    "$JSON_FILE_PATH" \
    "$REPORT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "推理完成！"
    echo "结果保存在: $REPORT_DIR"
else
    echo "推理失败！"
    exit 1
fi
