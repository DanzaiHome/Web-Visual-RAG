import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BochaConfig:
    api_key: str = os.getenv("BOCHA_API_KEY", "sk-XXX")
    api_url: str = os.getenv("BOCHA_API_URL", "https://api.bocha.cn/v1/web-search")
    timeout: int = _get_int_env("BOCHA_TIMEOUT", 20)


@dataclass(frozen=True)
class ChatAPIConfig:
    api_key: str = os.getenv(
        "CHAT_API_KEY",
        os.getenv("DASHSCOPE_API_KEY", "sk-XXX"),
    )
    api_base: str = os.getenv(
        "CHAT_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model: str = os.getenv("CHAT_MODEL", "qwen3.6-plus")
    api_mode: str = os.getenv("CHAT_API_MODE", "chat_completions")
    max_retries: int = _get_int_env("CHAT_MAX_RETRIES", 2)
    connect_timeout: int = _get_int_env("CHAT_CONNECT_TIMEOUT", 10)
    read_timeout: int = _get_int_env("CHAT_READ_TIMEOUT", 120)


@dataclass(frozen=True)
class ClipServerConfig:
    host: str = os.getenv("CLIP_SERVER_HOST", "127.0.0.1")
    port: int = _get_int_env("CLIP_SERVER_PORT", 8001)
    request_timeout: int = _get_int_env("CLIP_SERVER_REQUEST_TIMEOUT", 120)
    model_id: str = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    hf_endpoint: str = (
        os.getenv("CLIP_HF_ENDPOINT")
        or os.getenv("HF_ENDPOINT")
        or "https://hf-mirror.com"
    )
    local_model_dir: Path = Path(
        os.getenv(
            "CLIP_LOCAL_MODEL_DIR",
            str(PROJECT_ROOT / "models" / "clip-vit-base-patch32"),
        )
    )
    image_timeout: int = _get_int_env("CLIP_IMAGE_TIMEOUT", 20)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class WebFetchConfig:
    enabled: bool = _get_bool_env("WEB_FETCH_ENABLED", True)
    timeout: int = _get_int_env("WEB_FETCH_TIMEOUT", 18)
    max_retries: int = _get_int_env("WEB_FETCH_MAX_RETRIES", 2)
    cache_ttl_seconds: int = _get_int_env("WEB_FETCH_CACHE_TTL_SECONDS", 604800)
    max_bytes: int = _get_int_env("WEB_FETCH_MAX_BYTES", 2_500_000)
    min_text_chars: int = _get_int_env("WEB_FETCH_MIN_TEXT_CHARS", 300)
    cache_dir: Path = Path(
        os.getenv(
            "WEB_FETCH_CACHE_DIR",
            str(PROJECT_ROOT / ".cache" / "web_pages"),
        )
    )


BOCHA_CONFIG = BochaConfig()
CHAT_API_CONFIG = ChatAPIConfig()
CLIP_SERVER_CONFIG = ClipServerConfig()
WEB_FETCH_CONFIG = WebFetchConfig()
IMAGE_MATCH_THRESHOLD = float(os.getenv("IMAGE_MATCH_THRESHOLD", "0.5"))
