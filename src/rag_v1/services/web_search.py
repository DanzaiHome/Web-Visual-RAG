from typing import Any, Dict, List, Optional, Sequence

import requests

from rag_v1.config import BOCHA_CONFIG, WEB_FETCH_CONFIG
from rag_v1.services.web_page_fetcher import (
    FetchedPage,
    WebPageFetcher,
    canonicalize_url,
    is_valid_page_image_url,
)


class WebSearcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or BOCHA_CONFIG.api_key
        self.api_url = api_url or BOCHA_CONFIG.api_url
        self.timeout = timeout or BOCHA_CONFIG.timeout
        self.session = requests.Session()
        self.page_fetcher = WebPageFetcher()

        if not self.api_key:
            raise ValueError(
                "Bocha API key is required. Pass api_key=... or set BOCHA_API_KEY."
            )

    def search_engine(
        self,
        query: str,
        count: int = 10,
        summary: bool = True,
        freshness: str = "noLimit",
        include: Optional[str] = None,
        exclude: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if count < 1 or count > 50:
            raise ValueError("count must be between 1 and 50")

        payload: Dict[str, Any] = {
            "query": query,
            "freshness": freshness,
            "summary": summary,
            "count": count,
        }
        if include:
            payload["include"] = include
        if exclude:
            payload["exclude"] = exclude

        print(f"\nSearch url:\n{self.api_url}\n")
        response = self.session.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        try:
            result = response.json()
        except ValueError as exc:
            response.raise_for_status()
            raise RuntimeError("Bocha API did not return valid JSON") from exc

        if response.status_code != 200:
            message = result.get("message") or result.get("msg") or response.text
            log_id = result.get("log_id")
            raise RuntimeError(
                f"Bocha API request failed: status={response.status_code}, "
                f"message={message}, log_id={log_id}"
            )

        if result.get("code") != 200:
            raise RuntimeError(
                f"Bocha API returned error code={result.get('code')}, "
                f"msg={result.get('msg')}, log_id={result.get('log_id')}"
            )

        return result

    @staticmethod
    def _is_page_image_url(image_url: str) -> bool:
        return is_valid_page_image_url(image_url)

    @staticmethod
    def _group_images_by_host(
        image_items: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for item in image_items:
            host_page_url = str(item.get("hostPageUrl") or "").strip()
            content_url = str(item.get("contentUrl") or "").strip()
            thumbnail_url = str(item.get("thumbnailUrl") or "").strip()
            image_url = content_url or thumbnail_url

            if not host_page_url or not image_url:
                continue
            if not WebSearcher._is_page_image_url(image_url):
                continue

            host_key = canonicalize_url(host_page_url) or host_page_url
            grouped.setdefault(host_key, [])
            if image_url not in grouped[host_key]:
                grouped[host_key].append(image_url)

        return grouped

    @staticmethod
    def _merge_image_urls(limit: int, *image_url_groups: Sequence[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for image_urls in image_url_groups:
            for image_url in image_urls:
                normalized = str(image_url or "").strip()
                if not normalized or normalized in seen:
                    continue
                if not WebSearcher._is_page_image_url(normalized):
                    continue
                seen.add(normalized)
                merged.append(normalized)
                if len(merged) >= limit:
                    return merged
        return merged

    @staticmethod
    def _page_text_is_better(page: FetchedPage, fallback_text: str) -> bool:
        page_text = str(page.text or "").strip()
        fallback_text = str(fallback_text or "").strip()
        if not page_text:
            return False
        if fallback_text and page.is_probable_listing:
            if page.quality_score < 0.55 or len(page_text) < max(1200, int(len(fallback_text) * 1.2)):
                return False
        if fallback_text and page.quality_score < 0.2:
            return False
        if len(page_text) >= WEB_FETCH_CONFIG.min_text_chars:
            return True
        return len(page_text) > max(120, int(len(fallback_text) * 1.5))

    @staticmethod
    def _apply_fetched_page(
        doc: Dict[str, Any],
        page: FetchedPage,
        content_preview_len: int,
        top_n_images: int,
    ) -> None:
        fallback_text = str(doc.get("full_content") or doc.get("content") or "").strip()
        page_text = str(page.text or "").strip()

        doc["canonical_url"] = page.canonical_url or doc.get("canonical_url") or canonicalize_url(doc.get("url"))
        doc["web_fetch_status"] = page.fetch_status
        doc["web_fetch_error"] = page.error
        doc["web_fetch_from_cache"] = page.from_cache
        doc["web_fetch_final_url"] = page.final_url
        doc["web_fetch_status_code"] = page.status_code
        doc["web_fetch_content_type"] = page.content_type
        doc["web_fetch_content_hash"] = page.content_hash
        doc["web_fetch_extraction_method"] = page.extraction_method
        doc["web_fetch_quality_score"] = page.quality_score
        doc["web_fetch_is_probable_listing"] = page.is_probable_listing

        if page.title:
            current_title = str(doc.get("name") or "").strip()
            if not current_title or len(page.title) >= min(len(current_title), 24):
                doc["name"] = page.title
        if page.site_name and not doc.get("site_name"):
            doc["site_name"] = page.site_name
        if page.date_published and not doc.get("date_published"):
            doc["date_published"] = page.date_published
        if page.language and not doc.get("language"):
            doc["language"] = page.language

        doc["image_urls"] = WebSearcher._merge_image_urls(
            top_n_images,
            doc.get("image_urls") or [],
            page.image_urls,
        )

        if page.ok and WebSearcher._page_text_is_better(page, fallback_text):
            doc["content"] = page_text[:content_preview_len]
            doc["full_content"] = page_text
            doc["content_source"] = "web_page"
            return

        doc["content"] = fallback_text[:content_preview_len]
        doc["full_content"] = fallback_text
        doc["content_source"] = "bocha_summary_fallback"

    def search(
        self,
        query: str,
        candidate_k: int = 10,
        summary: bool = True,
        freshness: str = "noLimit",
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        top_n_images: int = 3,
        content_preview_len: int = 800,
        fetch_pages: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        result = self.search_engine(
            query=query,
            count=candidate_k,
            summary=summary,
            freshness=freshness,
            include=include,
            exclude=exclude,
        )

        data = result.get("data") or {}
        web_pages = (data.get("webPages") or {}).get("value") or []
        image_items = (data.get("images") or {}).get("value") or []
        images_by_host = self._group_images_by_host(image_items)
        docs: List[Dict[str, Any]] = []
        seen_urls = set()
        should_fetch_pages = WEB_FETCH_CONFIG.enabled if fetch_pages is None else fetch_pages

        for item in web_pages:
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            canonical_url = canonicalize_url(url)
            dedupe_key = canonical_url or url
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)

            summary_text = str(item.get("summary") or "").strip()
            snippet_text = str(item.get("snippet") or "").strip()
            content = summary_text or snippet_text

            image_urls = images_by_host.get(dedupe_key, [])[:top_n_images]

            doc = {
                "id": item.get("id"),
                "name": item.get("name"),
                "url": url,
                "canonical_url": canonical_url,
                "display_url": item.get("displayUrl"),
                "content": content[:content_preview_len],
                "full_content": content,
                "snippet": snippet_text,
                "summary": summary_text,
                "bocha_content": content,
                "content_source": "bocha_summary",
                "site_name": item.get("siteName"),
                "site_icon": item.get("siteIcon"),
                "date_published": item.get("datePublished"),
                "date_last_crawled": item.get("dateLastCrawled"),
                "cached_page_url": item.get("cachedPageUrl"),
                "language": item.get("language"),
                "is_family_friendly": item.get("isFamilyFriendly"),
                "is_navigational": item.get("isNavigational"),
                "image_urls": image_urls,
                "web_fetch_status": "disabled",
                "web_fetch_error": "",
                "web_fetch_from_cache": False,
                "web_fetch_quality_score": 0.0,
            }

            if should_fetch_pages:
                try:
                    fetched_page = self.page_fetcher.fetch(url)
                    self._apply_fetched_page(
                        doc=doc,
                        page=fetched_page,
                        content_preview_len=content_preview_len,
                        top_n_images=top_n_images,
                    )
                except Exception as exc:
                    doc["web_fetch_status"] = "fetcher_error"
                    doc["web_fetch_error"] = str(exc)
                    doc["content_source"] = "bocha_summary_fallback"

            docs.append(doc)

        return docs
