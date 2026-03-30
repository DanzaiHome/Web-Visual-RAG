```bash
CUDA_VISIBLE_DEVICES=0 \
uvicorn server:app --host 127.0.0.1 --port 18000 > "/remote-home1/xzhe/projects/CV_project/RAG_identifier/logs/RAG_server.log" 2>&1 &
```
