pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

cd src/rlvr
pip install -e ".[dev]"

# Addtional modules
pip install trl==0.17.0
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils
pip install math_verify
pip install flash-attn --no-build-isolation
pip3 install deepspeed

# vLLM support 
pip install vllm==0.7.2