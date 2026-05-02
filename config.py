import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from rag_v1.config import (
    BOCHA_CONFIG,
    CHAT_API_CONFIG,
    CLIP_SERVER_CONFIG,
    IMAGE_MATCH_THRESHOLD,
    PROJECT_ROOT,
)


PROJECT_DIR = PROJECT_ROOT
