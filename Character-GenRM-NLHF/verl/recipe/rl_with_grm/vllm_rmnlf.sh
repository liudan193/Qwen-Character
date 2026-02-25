conda activate verl_rm_nlhf

CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "/path/to/Tongyi-ConvAI/RM-NLHF-Qwen-7B" \
  --served-model-name "grm_rmnlhf" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8011 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.7