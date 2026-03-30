from huggingface_hub import snapshot_download

from pathlib import Path

local_dir = snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir= Path(__file__).resolve().parent / "Qwen3-VL-8B-Instruct",
)

# local_dir = snapshot_download(
#     repo_id="Qwen/Qwen3-VL-2B-Instruct",
#     local_dir= Path(__file__).resolve().parent / "Qwen3-VL-2B-Instruct",
# )

print(local_dir)