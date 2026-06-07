```bash
python val/eval_baseline.py --mode no_rag --trails 3
python val/eval_baseline.py --mode text_only_rag --trails 3
python val/eval_baseline.py --mode full_rag --multimodal-text-weight 0.7 --trails 3
python val/eval_baseline.py --mode route_router --trails 3
python val/eval_baseline.py --mode full_rag --multimodal-text-weight 0.7 --trails 3 --limit 22
python val/eval_baseline.py --mode full_rag --multimodal-text-weight 0.7 --trails 1 --limit 22
python val/eval_baseline.py --mode route_router --trails 3 --offset 22

python val/eval_baseline.py --mode no_rag --trails 1 --model-id qwen3-vl-plus-2025-12-19
python val/eval_baseline.py --mode full_rag --trails 1 --model-id qwen3-vl-30b-a3b-instruct
python val/eval_baseline.py --mode full_rag --trails 1 --model-id qwen3-vl-8b-instruct --web-search-provider bocha
python val/eval_baseline.py --mode full_rag --trails 1 --model-id qwen3-vl-8b-instruct --web-search-provider serpapi
```