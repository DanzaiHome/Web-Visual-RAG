from huggingface_hub import snapshot_download

from pathlib import Path

IDENTIFIER_DIR = Path(__file__).resolve().parent

snapshot_download(
    repo_id="google/siglip-so400m-patch14-384",
    local_dir=IDENTIFIER_DIR / "model",
    local_dir_use_symlinks=False
)