from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import requests

from rag_v1.config import PROJECT_ROOT, WEB_FETCH_CONFIG


CACHE_SCHEMA_VERSION = 8

TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "gbraid",
    "wbraid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "spm",
    "from",
    "ref",
    "source",
}

SUPPORTED_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)

IMAGE_FILE_EXTENSIONS = {
    ".apng",
    ".avif",
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
NON_IMAGE_FILE_EXTENSIONS = {
    ".7z",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".eot",
    ".html",
    ".js",
    ".json",
    ".map",
    ".otf",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".ttf",
    ".txt",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}
IGNORED_IMAGE_TOKENS = (
    "favicon",
    "siteicon",
    "site_icon",
    "qrcode",
    "barcode",
    "background",
    "bgr_",
    "bg_",
    "/bg",
    "button",
    "channel_tag",
    "navleft",
    "navright",
    "/qr",
    "-qr",
    "_qr",
    "/icon",
    "-icon",
    "_icon",
    "icon.",
    "sprite",
    "logo",
    "blank.",
    "spacer.",
    "pixel.",
    "tracking",
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 RAGFetcher/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.3",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "form",
    "button",
    "select",
    "option",
    "nav",
    "footer",
    "aside",
}

BLOCK_TAGS = {
    "address",
    "article",
    "blockquote",
    "dd",
    "details",
    "dialog",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}

CONTENT_ATTR_RE = re.compile(
    r"(^|[-_\s])("
    r"article|body|content|detail|entry|main|news|page|post|read|story|text"
    r")([-_\s]|$)",
    re.IGNORECASE,
)
NOISE_ATTR_RE = re.compile(
    r"(^|[-_\s])("
    r"ad|ads|advert|app|banner|breadcrumb|comment|cookie|copyright|"
    r"login|menu|modal|nav|popup|promo|recommend|related|share|sidebar|"
    r"social|subscribe|toolbar|widget"
    r")([-_\s]|$)",
    re.IGNORECASE,
)
HIDDEN_ATTR_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE)
IMAGE_ATTR_NOISE_RE = re.compile(
    r"(^|[-_\s])("
    r"ad|advert|avatar|background|banner|button|icon|logo|nav|qrcode|"
    r"share|social|sprite|tracking"
    r")([-_\s]|$)",
    re.IGNORECASE,
)

BOILERPLATE_PATTERNS = (
    r"all rights reserved",
    r"copyright",
    r"privacy policy",
    r"terms of (use|service)",
    r"cookie",
    r"subscribe",
    r"sign in",
    r"log in",
    r"follow us",
    r"share this",
    r"advertisement",
    r"sponsored",
    r"read more",
    r"recommended",
    r"related articles",
    r"版权",
    r"版权声明",
    r"免责声明",
    r"隐私政策",
    r"用户协议",
    r"广告",
    r"登录",
    r"注册",
    r"分享",
    r"关注我们",
    r"相关推荐",
    r"相关阅读",
    r"推荐阅读",
    r"扫码",
    r"微信",
    r"客服",
    r"备案号",
    r"责任编辑",
)
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


@dataclass
class FetchedPage:
    requested_url: str
    canonical_url: str
    final_url: str = ""
    status_code: Optional[int] = None
    ok: bool = False
    fetch_status: str = "failed"
    error: str = ""
    title: str = ""
    text: str = ""
    excerpt: str = ""
    site_name: str = ""
    date_published: str = ""
    language: str = ""
    image_urls: List[str] = field(default_factory=list)
    content_type: str = ""
    fetched_at: str = ""
    from_cache: bool = False
    content_hash: str = ""
    extraction_method: str = ""
    quality_score: float = 0.0
    is_probable_listing: bool = False


@dataclass
class _TextBlock:
    text: str
    score: float
    tag: str


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonicalize_url(url: object) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    if raw_url.startswith("//"):
        raw_url = f"https:{raw_url}"

    parsed = urlsplit(raw_url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    if not parsed.hostname:
        return ""

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower().rstrip(".")
    if host.startswith("m."):
        host = f"www.{host[2:]}"

    try:
        port = parsed.port
    except ValueError:
        return ""
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{netloc}:{port}"

    path = parsed.path or ""
    path = quote(unquote(path), safe="/:@-._~!$&'()*+,;=")
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""

    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        lowered_key = key.lower()
        if lowered_key in TRACKING_QUERY_KEYS:
            continue
        if any(lowered_key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_pairs.append((key, value))
    query = urlencode(sorted(query_pairs), doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def _attrs_to_dict(attrs: Sequence[Tuple[str, Optional[str]]]) -> Dict[str, str]:
    return {key.lower(): (value or "") for key, value in attrs}


def _attr_text(attrs: Dict[str, str]) -> str:
    return " ".join(
        value
        for key, value in attrs.items()
        if key in {"id", "class", "role", "aria-label", "itemprop"}
    )


def _has_noise_attrs(attrs: Dict[str, str]) -> bool:
    attr_text = _attr_text(attrs)
    if not attr_text:
        return False
    if CONTENT_ATTR_RE.search(attr_text):
        return False
    return bool(NOISE_ATTR_RE.search(attr_text))


def _has_hidden_attrs(attrs: Dict[str, str]) -> bool:
    if attrs.get("hidden") is not None and "hidden" in attrs:
        return True
    style = attrs.get("style", "")
    return bool(style and HIDDEN_ATTR_RE.search(style))


def _clean_text(text: object) -> str:
    value = unescape(str(text or ""))
    value = value.replace("\xa0", " ")
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", value)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _text_signature(text: object, max_len: int = 1000) -> str:
    return "".join(ch.lower() for ch in str(text or "") if ch.isalnum())[:max_len]


def _looks_like_boilerplate(text: str) -> bool:
    lowered = text.lower()
    if BOILERPLATE_RE.search(lowered) and len(text) < 220:
        return True
    if len(text) <= 18 and re.search(r"(home|menu|search|login|登录|首页|更多|返回)", lowered):
        return True
    if text.count("|") >= 4 or text.count(" / ") >= 4:
        return True
    alpha_num = sum(1 for ch in text if ch.isalnum())
    if text and alpha_num / max(1, len(text)) < 0.35:
        return True
    return False


def _is_useful_block(block: _TextBlock) -> bool:
    text = block.text
    if len(text) < 12:
        return False
    if _looks_like_boilerplate(text):
        return False
    if len(text) < 30 and block.tag not in {"h1", "h2", "h3", "h4", "li"}:
        return False
    return True


def _dedupe_blocks(blocks: Iterable[_TextBlock]) -> List[_TextBlock]:
    deduped: List[_TextBlock] = []
    signatures: List[str] = []

    for block in blocks:
        signature = _text_signature(block.text)
        if not signature:
            continue
        if signature in signatures:
            continue
        if len(signature) >= 80 and any(
            SequenceMatcher(None, signature, existing).ratio() >= 0.96
            for existing in signatures
            if len(existing) >= 80
        ):
            continue
        signatures.append(signature)
        deduped.append(block)

    return deduped


def _is_article_body_block(block: _TextBlock) -> bool:
    return block.tag in {"p", "li", "blockquote", "pre"} and len(block.text) >= 120


def _first_non_empty(*values: object) -> str:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return ""


def _strip_title_suffix(title: str, site_name: str) -> str:
    title = _clean_text(title)
    site_name = _clean_text(site_name)
    if not title or not site_name:
        return title

    separators = (" - ", " | ", "_", " -- ")
    for separator in separators:
        suffix = f"{separator}{site_name}"
        if title.endswith(suffix):
            return title[: -len(suffix)].strip()
    return title


def _extract_json_ld_date(html: str) -> str:
    match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html, re.IGNORECASE)
    if match:
        return _clean_text(match.group(1))
    match = re.search(r'"pubdate"\s*:\s*"([^"]+)"', html, re.IGNORECASE)
    if match:
        return _clean_text(match.group(1))
    return ""


def _best_image_from_srcset(srcset: str) -> str:
    candidates = []
    for part in srcset.split(","):
        tokens = part.strip().split()
        if tokens:
            candidates.append(tokens[0])
    return candidates[-1] if candidates else ""


def _url_path_extension(path: str) -> str:
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return ""
    suffix = filename.rsplit(".", 1)[-1]
    if not suffix or len(suffix) > 8:
        return ""
    return f".{suffix.lower()}"


def is_valid_page_image_url(image_url: object) -> bool:
    parsed = urlsplit(str(image_url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    path = unquote(parsed.path).lower()
    query = parsed.query.lower()
    if any(token in path or token in query for token in IGNORED_IMAGE_TOKENS):
        return False

    extension = _url_path_extension(path)
    if extension in NON_IMAGE_FILE_EXTENSIONS or extension in {".ico", ".svg", ".gif"}:
        return False
    if extension and extension not in IMAGE_FILE_EXTENSIONS:
        return False
    return True


def _dimension_from_attr(value: str) -> Optional[int]:
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _image_dimensions_are_reasonable(attrs: Dict[str, str]) -> bool:
    width = _dimension_from_attr(
        attrs.get("width") or attrs.get("data-width") or attrs.get("naturalwidth") or ""
    )
    height = _dimension_from_attr(
        attrs.get("height") or attrs.get("data-height") or attrs.get("naturalheight") or ""
    )
    if width is not None and height is not None:
        return width >= 80 and height >= 80
    if width is not None:
        return width >= 48
    if height is not None:
        return height >= 48
    return True


def _image_attrs_are_content_like(attrs: Dict[str, str]) -> bool:
    attr_text = " ".join(
        str(attrs.get(key) or "")
        for key in (
            "id",
            "class",
            "role",
            "alt",
            "title",
            "aria-label",
            "src",
            "data-src",
            "data-original",
            "data-lazy-src",
        )
    )
    return not IMAGE_ATTR_NOISE_RE.search(attr_text)


class _ReadableHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.base_href = ""
        self.blocks: List[_TextBlock] = []
        self.meta: Dict[str, str] = {}
        self.images: List[str] = []
        self.canonical_href = ""
        self.title_parts: List[str] = []
        self.h1_parts: List[str] = []
        self.time_values: List[str] = []
        self.html_lang = ""
        self._stack: List[Tuple[str, Dict[str, str]]] = []
        self._block_parts: List[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._in_h1 = False

    def handle_starttag(self, tag: str, attrs_list: Sequence[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attrs = _attrs_to_dict(attrs_list)

        if tag == "html" and attrs.get("lang"):
            self.html_lang = attrs["lang"].strip()
        if tag == "base" and attrs.get("href"):
            self.base_href = attrs["href"].strip()
        if tag == "meta":
            self._record_meta(attrs)

        if self._skip_depth:
            if tag not in VOID_TAGS:
                self._skip_depth += 1
            return

        if tag == "img":
            self._record_image(attrs)
        if tag == "time":
            datetime_value = attrs.get("datetime") or attrs.get("content")
            if datetime_value:
                self.time_values.append(datetime_value.strip())
        if tag == "link":
            rel = attrs.get("rel", "").lower()
            href = attrs.get("href", "")
            if href and "canonical" in rel and not self.canonical_href:
                self.canonical_href = urljoin(self.base_href or self.base_url, href.strip())
            is_preload_image = "preload" in rel and attrs.get("as", "").lower() == "image"
            if href and ("image_src" in rel or is_preload_image):
                self._append_image(href)

        skip_site_header = tag == "header" and not self._inside_content_container()
        if skip_site_header or tag in SKIP_TAGS or _has_hidden_attrs(attrs) or _has_noise_attrs(attrs):
            if tag not in VOID_TAGS:
                self._flush_block()
                self._skip_depth = 1
            return

        if tag in BLOCK_TAGS:
            self._flush_block()

        self._stack.append((tag, attrs))
        if tag == "title":
            self._in_title = True
        if tag == "h1":
            self._in_h1 = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if tag in BLOCK_TAGS:
            self._flush_block()

        if tag == "title":
            self._in_title = False
        if tag == "h1":
            self._in_h1 = False

        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                break

    def handle_data(self, data: str) -> None:
        text = _clean_text(data)
        if not text:
            return

        if self._in_title:
            self.title_parts.append(text)
            return
        if any(tag in {"head", "meta"} for tag, _ in self._stack):
            return
        if self._skip_depth:
            return

        if self._in_h1:
            self.h1_parts.append(text)
        self._block_parts.append(text)

    def close(self) -> None:
        self._flush_block()
        super().close()

    def _record_meta(self, attrs: Dict[str, str]) -> None:
        key = (
            attrs.get("property")
            or attrs.get("name")
            or attrs.get("itemprop")
            or attrs.get("http-equiv")
            or ""
        ).strip().lower()
        value = attrs.get("content", "").strip()
        if key and value and key not in self.meta:
            self.meta[key] = value
        if key in {"og:image", "twitter:image", "twitter:image:src"} and value:
            self._append_image(value)

    def _record_image(self, attrs: Dict[str, str]) -> None:
        if not _image_attrs_are_content_like(attrs):
            return
        if not _image_dimensions_are_reasonable(attrs):
            return
        image = (
            attrs.get("src")
            or attrs.get("data-src")
            or attrs.get("data-original")
            or attrs.get("data-lazy-src")
            or attrs.get("data-url")
            or ""
        )
        if not image and attrs.get("srcset"):
            image = _best_image_from_srcset(attrs["srcset"])
        if image:
            self._append_image(image)

    def _append_image(self, image_url: str) -> None:
        base = self.base_href or self.base_url
        absolute_url = urljoin(base, image_url.strip())
        if absolute_url and absolute_url not in self.images and is_valid_page_image_url(absolute_url):
            self.images.append(absolute_url)

    def _current_score(self) -> float:
        score = 0.0
        for tag, attrs in self._stack:
            if tag == "article":
                score += 3.0
            elif tag == "main":
                score += 2.5
            elif tag in {"p", "blockquote"}:
                score += 0.8
            elif tag in {"h1", "h2", "h3"}:
                score += 0.5

            attr_text = _attr_text(attrs)
            if CONTENT_ATTR_RE.search(attr_text):
                score += 2.0
            if NOISE_ATTR_RE.search(attr_text):
                score -= 3.0
        return score

    def _inside_content_container(self) -> bool:
        for tag, attrs in self._stack:
            if tag in {"article", "main"}:
                return True
            if CONTENT_ATTR_RE.search(_attr_text(attrs)):
                return True
        return False

    def _current_tag(self) -> str:
        for tag, _ in reversed(self._stack):
            if tag in BLOCK_TAGS:
                return tag
        return ""

    def _flush_block(self) -> None:
        if not self._block_parts:
            return
        text = _clean_text(" ".join(self._block_parts))
        self._block_parts = []
        if not text:
            return
        self.blocks.append(
            _TextBlock(
                text=text,
                score=self._current_score(),
                tag=self._current_tag(),
            )
        )


def _select_main_blocks(blocks: Sequence[_TextBlock]) -> List[_TextBlock]:
    useful_blocks = _dedupe_blocks(block for block in blocks if _is_useful_block(block))
    if not useful_blocks:
        return []

    best_score = max(block.score for block in useful_blocks)
    if best_score >= 2.0:
        score_floor = max(1.0, best_score - 2.0)
        body_floor = max(1.0, best_score - 4.0)
        selected = [
            block
            for block in useful_blocks
            if block.score >= score_floor
            or (_is_article_body_block(block) and block.score >= body_floor)
            or (block.tag in {"h1", "h2", "h3"} and block.score >= 0.5)
        ]
    else:
        selected = [
            block
            for block in useful_blocks
            if block.tag in {"p", "li", "blockquote", "pre", "h1", "h2", "h3"}
            or len(block.text) >= 80
        ]

    selected = _dedupe_blocks(selected)
    selected = _drop_related_title_blocks_after_body(selected)
    if len(" ".join(block.text for block in selected)) < 300:
        fallback = [block for block in useful_blocks if len(block.text) >= 40]
        if len(" ".join(block.text for block in fallback)) > len(
            " ".join(block.text for block in selected)
        ):
            selected = _dedupe_blocks(fallback)
            selected = _drop_related_title_blocks_after_body(selected)

    return selected


def _looks_like_title_list_item(text: object) -> bool:
    cleaned = _clean_text(text)
    if len(cleaned) < 8 or len(cleaned) > 180:
        return False
    if len(cleaned) <= 100:
        return True

    sentence_punctuation = sum(cleaned.count(char) for char in "。！？?!")
    comma_punctuation = sum(cleaned.count(char) for char in "，、；;,")
    return sentence_punctuation <= 1 and comma_punctuation <= 2


def _drop_related_title_blocks_after_body(blocks: Sequence[_TextBlock]) -> List[_TextBlock]:
    if not any(_is_article_body_block(block) for block in blocks):
        return list(blocks)

    pruned: List[_TextBlock] = []
    seen_body = False
    for block in blocks:
        if _is_article_body_block(block):
            seen_body = True
            pruned.append(block)
            continue
        if seen_body and block.tag in {"div", "h2", "h3", "h4", "li"}:
            if _looks_like_title_list_item(block.text):
                continue
        pruned.append(block)
    return pruned


def _listing_density(blocks: Sequence[_TextBlock]) -> float:
    if not blocks:
        return 0.0

    block_count = len(blocks)
    short_ratio = sum(1 for block in blocks if len(block.text) <= 120) / block_count
    long_ratio = sum(1 for block in blocks if len(block.text) >= 180) / block_count
    list_tag_ratio = sum(1 for block in blocks if block.tag in {"li", "h2", "h3", "h4"}) / block_count
    title_like_ratio = sum(1 for block in blocks if _looks_like_title_list_item(block.text)) / block_count
    density = 0.55 * short_ratio + 0.35 * title_like_ratio + 0.15 * list_tag_ratio - 0.5 * long_ratio
    return max(0.0, min(1.0, density))


def _is_probable_listing(blocks: Sequence[_TextBlock], text: str) -> bool:
    if len(blocks) < 4:
        return False
    lengths = [len(block.text) for block in blocks]
    avg_length = sum(lengths) / max(1, len(lengths))
    long_block_ratio = sum(1 for length in lengths if length >= 120) / max(1, len(lengths))
    density = _listing_density(blocks)
    if len(blocks) < 8:
        return density >= 0.82 and len(text) >= 250 and long_block_ratio == 0
    return (
        len(text) < 4000
        and avg_length < 95
        and long_block_ratio < 0.35
    ) or density >= 0.72


def _listing_quality_penalty(blocks: Sequence[_TextBlock], text: str) -> float:
    density = _listing_density(blocks)
    block_count_penalty = min(max(len(blocks) - 10, 0) / 80, 0.2)
    long_text_penalty = 0.08 if len(text) > 5000 and density >= 0.65 else 0.0
    return min(0.6, 0.2 + density * 0.35 + block_count_penalty + long_text_penalty)


def _quality_score(blocks: Sequence[_TextBlock], text: str, title: str, date_published: str) -> float:
    if not text:
        return 0.0
    length_score = min(len(text) / 3500, 1.0)
    paragraph_score = min(len([block for block in blocks if len(block.text) >= 80]) / 8, 1.0)
    metadata_score = 0.0
    if title:
        metadata_score += 0.08
    if date_published:
        metadata_score += 0.07
    replacement_penalty = min(text.count("\ufffd") / max(1, len(text)) * 4, 0.3)
    score = 0.65 * length_score + 0.25 * paragraph_score + metadata_score - replacement_penalty
    return max(0.0, min(1.0, score))


def extract_html_document(
    requested_url: str,
    final_url: str,
    html: str,
    content_type: str = "text/html",
    status_code: Optional[int] = None,
) -> FetchedPage:
    parser = _ReadableHTMLParser(final_url or requested_url)
    parser.feed(html)
    parser.close()
    canonical_url = (
        canonicalize_url(parser.canonical_href)
        or canonicalize_url(final_url)
        or canonicalize_url(requested_url)
    )

    site_name = _first_non_empty(parser.meta.get("og:site_name"), urlsplit(final_url).hostname)
    title = _first_non_empty(
        parser.meta.get("og:title"),
        parser.meta.get("twitter:title"),
        " ".join(parser.h1_parts),
        " ".join(parser.title_parts),
    )
    title = _strip_title_suffix(title, site_name)
    date_published = _first_non_empty(
        parser.meta.get("article:published_time"),
        parser.meta.get("datepublished"),
        parser.meta.get("date"),
        parser.meta.get("pubdate"),
        parser.meta.get("dc.date"),
        parser.meta.get("dc.date.issued"),
        parser.meta.get("publishdate"),
        parser.time_values[0] if parser.time_values else "",
        _extract_json_ld_date(html),
    )
    language = _first_non_empty(parser.meta.get("language"), parser.html_lang)

    selected_blocks = _select_main_blocks(parser.blocks)
    text = "\n\n".join(block.text for block in selected_blocks).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest() if text else ""
    listing = _is_probable_listing(selected_blocks, text)
    quality = _quality_score(selected_blocks, text, title, date_published)
    if listing:
        quality = max(0.0, quality - _listing_quality_penalty(selected_blocks, text))

    return FetchedPage(
        requested_url=requested_url,
        canonical_url=canonical_url,
        final_url=final_url or requested_url,
        status_code=status_code,
        ok=bool(text),
        fetch_status="success" if text else "empty_content",
        title=title,
        text=text,
        excerpt=text[:500],
        site_name=site_name,
        date_published=date_published,
        language=language,
        image_urls=parser.images[:20],
        content_type=content_type,
        fetched_at=_now_iso(),
        content_hash=content_hash,
        extraction_method="html_parser",
        quality_score=quality,
        is_probable_listing=listing,
    )


def extract_text_document(
    requested_url: str,
    final_url: str,
    text: str,
    content_type: str = "text/plain",
    status_code: Optional[int] = None,
) -> FetchedPage:
    paragraphs = []
    for line in text.splitlines():
        cleaned = _clean_text(line)
        if cleaned and not _looks_like_boilerplate(cleaned):
            paragraphs.append(cleaned)
    paragraphs = [block.text for block in _dedupe_blocks(_TextBlock(p, 0.0, "p") for p in paragraphs)]
    body = "\n\n".join(paragraphs).strip()
    return FetchedPage(
        requested_url=requested_url,
        canonical_url=canonicalize_url(final_url) or canonicalize_url(requested_url),
        final_url=final_url or requested_url,
        status_code=status_code,
        ok=bool(body),
        fetch_status="success" if body else "empty_content",
        text=body,
        excerpt=body[:500],
        content_type=content_type,
        fetched_at=_now_iso(),
        content_hash=hashlib.sha256(body.encode("utf-8", errors="ignore")).hexdigest() if body else "",
        extraction_method="plain_text",
        quality_score=min(len(body) / 3500, 1.0) if body else 0.0,
    )


class WebPageFetcher:
    def __init__(
        self,
        session: Optional[requests.Session] = None,
        cache_dir: Optional[Path] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        cache_ttl_seconds: Optional[int] = None,
        max_bytes: Optional[int] = None,
        min_text_chars: Optional[int] = None,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout or WEB_FETCH_CONFIG.timeout
        self.max_retries = WEB_FETCH_CONFIG.max_retries if max_retries is None else max_retries
        self.cache_ttl_seconds = (
            WEB_FETCH_CONFIG.cache_ttl_seconds
            if cache_ttl_seconds is None
            else cache_ttl_seconds
        )
        self.max_bytes = max_bytes or WEB_FETCH_CONFIG.max_bytes
        self.min_text_chars = min_text_chars or WEB_FETCH_CONFIG.min_text_chars
        self.cache_dir = (cache_dir or WEB_FETCH_CONFIG.cache_dir).resolve()
        self._validate_cache_dir()

    def fetch(self, url: object) -> FetchedPage:
        requested_url = str(url or "").strip()
        canonical_url = canonicalize_url(requested_url)
        if not canonical_url:
            return self._failed_page(
                requested_url=requested_url,
                canonical_url="",
                status="invalid_url",
                error="unsupported or invalid URL",
            )

        cached_page = self._read_cache(canonical_url)
        if cached_page is not None:
            return cached_page

        max_attempts = max(1, self.max_retries + 1)
        last_error = ""
        for attempt in range(max_attempts):
            response: Optional[requests.Response] = None
            try:
                response = self.session.get(
                    canonical_url,
                    headers=DEFAULT_HEADERS,
                    timeout=(min(6, self.timeout), self.timeout),
                    allow_redirects=True,
                    stream=True,
                )
                status_code = response.status_code
                if status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                if status_code >= 400:
                    page = self._failed_page(
                        requested_url=requested_url,
                        canonical_url=canonical_url,
                        final_url=response.url,
                        status="http_error",
                        error=f"HTTP {status_code}",
                        status_code=status_code,
                        content_type=response.headers.get("Content-Type", ""),
                    )
                    self._write_cache(canonical_url, page)
                    return page

                content_type = response.headers.get("Content-Type", "")
                if not self._is_supported_content_type(content_type):
                    page = self._failed_page(
                        requested_url=requested_url,
                        canonical_url=canonical_url,
                        final_url=response.url,
                        status="unsupported_content_type",
                        error=f"unsupported content type: {content_type}",
                        status_code=status_code,
                        content_type=content_type,
                    )
                    self._write_cache(canonical_url, page)
                    return page

                body = self._read_limited_response(response)
                decoded = self._decode_response(response, body)
                if self._is_plain_text(content_type):
                    page = extract_text_document(
                        requested_url=requested_url,
                        final_url=response.url,
                        text=decoded,
                        content_type=content_type,
                        status_code=status_code,
                    )
                else:
                    page = extract_html_document(
                        requested_url=requested_url,
                        final_url=response.url,
                        html=decoded,
                        content_type=content_type,
                        status_code=status_code,
                    )

                if page.ok and len(page.text) < self.min_text_chars:
                    page.fetch_status = "short_content"
                self._write_cache(canonical_url, page)
                return page
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = str(exc)
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 4))
                    continue
            finally:
                if response is not None:
                    response.close()

        page = self._failed_page(
            requested_url=requested_url,
            canonical_url=canonical_url,
            status="request_failed",
            error=last_error or "request failed",
        )
        self._write_cache(canonical_url, page)
        return page

    def _validate_cache_dir(self) -> None:
        project_root = PROJECT_ROOT.resolve()
        if not _is_relative_to(self.cache_dir, project_root):
            raise ValueError(f"WEB_FETCH_CACHE_DIR must be under project root: {project_root}")

    def _cache_path(self, canonical_url: str) -> Path:
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, canonical_url: str) -> Optional[FetchedPage]:
        if self.cache_ttl_seconds <= 0:
            return None
        path = self._cache_path(canonical_url)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError):
            return None

        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        try:
            cached_at = float(payload.get("cached_at_epoch") or 0)
        except (TypeError, ValueError):
            return None
        if cached_at and time.time() - cached_at > self.cache_ttl_seconds:
            return None
        page_payload = payload.get("page")
        if not isinstance(page_payload, dict):
            return None
        try:
            page = FetchedPage(**page_payload)
        except TypeError:
            return None
        page.from_cache = True
        return page

    def _write_cache(self, canonical_url: str, page: FetchedPage) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(canonical_url)
        tmp_path = path.with_suffix(".json.tmp")
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cached_at_epoch": time.time(),
            "canonical_url": canonical_url,
            "page": asdict(page),
        }
        try:
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
            tmp_path.replace(path)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def _read_limited_response(self, response: requests.Response) -> bytes:
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_bytes:
                overflow = total - self.max_bytes
                chunks.append(chunk[:-overflow] if overflow < len(chunk) else b"")
                break
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode_response(response: requests.Response, body: bytes) -> str:
        content_type = response.headers.get("Content-Type", "").lower()
        header_has_charset = "charset=" in content_type
        encoding = response.encoding if header_has_charset else None
        if not encoding:
            encoding = WebPageFetcher._encoding_from_meta(body)
        if not encoding:
            encoding = response.apparent_encoding or "utf-8"
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")

    @staticmethod
    def _encoding_from_meta(body: bytes) -> str:
        head = body[:4096].decode("ascii", errors="ignore")
        match = re.search(
            r"<meta[^>]+charset=[\"']?\s*([a-zA-Z0-9._-]+)",
            head,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        match = re.search(
            r"<meta[^>]+content=[\"'][^\"']*charset=([a-zA-Z0-9._-]+)",
            head,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _is_supported_content_type(content_type: str) -> bool:
        normalized = content_type.lower().split(";", 1)[0].strip()
        if not normalized:
            return True
        return normalized in SUPPORTED_CONTENT_TYPES

    @staticmethod
    def _is_plain_text(content_type: str) -> bool:
        return content_type.lower().split(";", 1)[0].strip() == "text/plain"

    @staticmethod
    def _failed_page(
        requested_url: str,
        canonical_url: str,
        status: str,
        error: str,
        final_url: str = "",
        status_code: Optional[int] = None,
        content_type: str = "",
    ) -> FetchedPage:
        return FetchedPage(
            requested_url=requested_url,
            canonical_url=canonical_url,
            final_url=final_url or requested_url,
            status_code=status_code,
            ok=False,
            fetch_status=status,
            error=error,
            content_type=content_type,
            fetched_at=_now_iso(),
        )
