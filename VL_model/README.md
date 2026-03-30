```bash
CUDA_VISIBLE_DEVICES=1 \
vllm serve /remote-home1/xzhe/projects/CV_project/VL_model/Qwen3-VL-2B-Instruct \
    --host 127.0.0.1 \
    --port 18001 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --served-model-name Qwen3-VL > "/remote-home1/xzhe/projects/CV_project/VL_model/logs/VL_server.log" 2>&1 &
```


```bash
CUDA_VISIBLE_DEVICES=1 \
vllm serve /remote-home1/xzhe/projects/CV_project/VL_model/Qwen3-VL-8B-Instruct \
    --host 127.0.0.1 \
    --port 18001 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --served-model-name Qwen3-VL > "/remote-home1/xzhe/projects/CV_project/VL_model/logs/VL_8B_server.log" 2>&1 &
```