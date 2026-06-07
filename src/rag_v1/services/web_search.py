import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from rag_v1.config import BOCHA_CONFIG, SERPAPI_CONFIG, WEB_FETCH_CONFIG, WEB_SEARCH_PROVIDER
from rag_v1.services.web_page_fetcher import (
    FetchedPage,
    WebPageFetcher,
    canonicalize_url,
    is_valid_page_image_url,
)
from rag_v1.timing import get_active_timing


class WebSearcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.provider = (provider or WEB_SEARCH_PROVIDER or "serpapi").strip().lower()
        if self.provider not in {"serpapi", "bocha"}:
            raise ValueError(
                f"Unsupported web search provider: {self.provider!r}. "
                "Expected 'serpapi' or 'bocha'."
            )

        if self.provider == "bocha":
            self.api_key = api_key or BOCHA_CONFIG.api_key
            self.api_url = api_url or BOCHA_CONFIG.api_url
            self.timeout = timeout or BOCHA_CONFIG.timeout
            self.engine = ""
            self.google_domain = ""
            self.gl = ""
            self.hl = ""
            self.location = ""
            self.no_cache = False
        else:
            self.api_key = api_key or SERPAPI_CONFIG.api_key
            self.api_url = api_url or SERPAPI_CONFIG.api_url
            self.timeout = timeout or SERPAPI_CONFIG.timeout
            self.engine = SERPAPI_CONFIG.engine
            self.google_domain = SERPAPI_CONFIG.google_domain
            self.gl = SERPAPI_CONFIG.gl
            self.hl = SERPAPI_CONFIG.hl
            self.location = SERPAPI_CONFIG.location
            self.no_cache = SERPAPI_CONFIG.no_cache

        self.session = requests.Session()
        self.page_fetcher = WebPageFetcher()

        if not self.api_key:
            env_name = "BOCHA_API_KEY" if self.provider == "bocha" else "SERPAPI_API_KEY"
            provider_name = "Bocha" if self.provider == "bocha" else "SerpApi"
            raise ValueError(
                f"{provider_name} key is required. Pass api_key=... or set {env_name}."
            )

    @staticmethod
    def _serpapi_tbs_from_freshness(freshness: str) -> Optional[str]:
        mapping = {
            "oneDay": "qdr:d",
            "oneWeek": "qdr:w",
            "oneMonth": "qdr:m",
            "oneYear": "qdr:y",
        }
        return mapping.get(str(freshness or "").strip())

    @staticmethod
    def _is_no_results_error(message: object) -> bool:
        normalized = str(message or "").strip().lower()
        if not normalized:
            return False
        return (
            "hasn't returned any results for this query" in normalized
            or "didn't return any results for this query" in normalized
            or "no results" in normalized
        )

    def _search_engine_bocha(
        self,
        query: str,
        count: int,
        summary: bool,
        freshness: str,
        include: Optional[str],
        exclude: Optional[str],
        debug: bool,
    ) -> Dict[str, Any]:
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

        if debug:
            print(f"\nSearch provider:\n{self.provider}\n")
            print(f"Search url:\n{self.api_url}\n")

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

    def _search_engine_serpapi(
        self,
        query: str,
        count: int,
        summary: bool,
        freshness: str,
        include: Optional[str],
        exclude: Optional[str],
        debug: bool,
    ) -> Dict[str, Any]:
        del summary

        if count < 1 or count > 100:
            raise ValueError("count must be between 1 and 100")

        effective_query = query.strip()
        if include:
            effective_query = f"{effective_query} {include}".strip()
        if exclude:
            effective_query = f"{effective_query} {exclude}".strip()

        params: Dict[str, Any] = {
            "engine": self.engine,
            "q": effective_query,
            "api_key": self.api_key,
            "google_domain": self.google_domain,
            "gl": self.gl,
            "hl": self.hl,
            "num": count,
            "output": "json",
        }
        if self.location:
            params["location"] = self.location
        if self.no_cache:
            params["no_cache"] = "true"
        tbs = self._serpapi_tbs_from_freshness(freshness)
        if tbs:
            params["tbs"] = tbs

        if debug:
            print(f"\nSearch provider:\n{self.provider}\n")
            print(f"Search url:\n{self.api_url}\n")

        response = self.session.get(
            self.api_url,
            params=params,
            timeout=self.timeout,
        )

        try:
            result = response.json()
        except ValueError as exc:
            response.raise_for_status()
            raise RuntimeError("SerpApi did not return valid JSON") from exc

        if response.status_code != 200:
            message = result.get("error") or response.text
            raise RuntimeError(
                f"SerpApi request failed: status={response.status_code}, message={message}"
            )

        if result.get("error"):
            if self._is_no_results_error(result.get("error")):
                return {"organic_results": []}
            raise RuntimeError(f"SerpApi returned error: {result.get('error')}")

        return result

    def search_engine(
        self,
        query: str,
        count: int = 10,
        summary: bool = True,
        freshness: str = "noLimit",
        include: Optional[str] = None,
        exclude: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")

        start_time = time.perf_counter()
        try:
            if self.provider == "bocha":
                return self._search_engine_bocha(
                    query=query,
                    count=count,
                    summary=summary,
                    freshness=freshness,
                    include=include,
                    exclude=exclude,
                    debug=debug,
                )
            return self._search_engine_serpapi(
                query=query,
                count=count,
                summary=summary,
                freshness=freshness,
                include=include,
                exclude=exclude,
                debug=debug,
            )
        finally:
            timing = get_active_timing()
            if timing is not None:
                timing.add("web_search_api", time.perf_counter() - start_time)

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
    def _organic_result_images(item: Dict[str, Any]) -> List[str]:
        image_urls: List[str] = []
        for key in ("thumbnail",):
            value = str(item.get(key) or "").strip()
            if value and WebSearcher._is_page_image_url(value) and value not in image_urls:
                image_urls.append(value)
        return image_urls

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

    def _fallback_content_source(self) -> str:
        return "bocha_summary_fallback" if self.provider == "bocha" else "search_result_snippet_fallback"

    def _initial_content_source(self) -> str:
        return "bocha_summary" if self.provider == "bocha" else "search_result_snippet"

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

    def _search_bocha_docs(
        self,
        result: Dict[str, Any],
        top_n_images: int,
        content_preview_len: int,
        fetch_pages: Optional[bool],
    ) -> List[Dict[str, Any]]:
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
                "content_source": self._initial_content_source(),
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
                    if doc.get("content_source") != "web_page":
                        doc["content"] = content[:content_preview_len]
                        doc["full_content"] = content
                        doc["content_source"] = self._fallback_content_source()
                except Exception as exc:
                    doc["web_fetch_status"] = "fetcher_error"
                    doc["web_fetch_error"] = str(exc)
                    doc["content_source"] = self._fallback_content_source()

            docs.append(doc)

        return docs

    def _search_serpapi_docs(
        self,
        result: Dict[str, Any],
        top_n_images: int,
        content_preview_len: int,
        summary: bool,
        fetch_pages: Optional[bool],
    ) -> List[Dict[str, Any]]:
        web_pages = result.get("organic_results") or []
        docs: List[Dict[str, Any]] = []
        seen_urls = set()
        should_fetch_pages = WEB_FETCH_CONFIG.enabled if fetch_pages is None else fetch_pages

        for item in web_pages:
            url = str(item.get("link") or "").strip()
            if not url:
                continue
            canonical_url = canonicalize_url(url)
            dedupe_key = canonical_url or url
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)

            summary_text = str(item.get("snippet") or "").strip() if summary else ""
            snippet_text = str(item.get("snippet") or "").strip()
            content = summary_text or snippet_text
            image_urls = self._organic_result_images(item)[:top_n_images]

            doc = {
                "id": item.get("position"),
                "name": item.get("title"),
                "url": url,
                "canonical_url": canonical_url,
                "display_url": item.get("displayed_link"),
                "content": content[:content_preview_len],
                "full_content": content,
                "snippet": snippet_text,
                "summary": summary_text,
                "content_source": self._initial_content_source(),
                "site_name": item.get("source"),
                "site_icon": item.get("favicon"),
                "date_published": item.get("date"),
                "cached_page_url": item.get("cached_page_link"),
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
                    if doc.get("content_source") != "web_page":
                        doc["content"] = content[:content_preview_len]
                        doc["full_content"] = content
                        doc["content_source"] = self._fallback_content_source()
                except Exception as exc:
                    doc["web_fetch_status"] = "fetcher_error"
                    doc["web_fetch_error"] = str(exc)
                    doc["content_source"] = self._fallback_content_source()

            docs.append(doc)

        return docs

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
        debug: bool = False,
        fetch_pages: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        result = self.search_engine(
            query=query,
            count=candidate_k,
            summary=summary,
            freshness=freshness,
            include=include,
            exclude=exclude,
            debug=debug,
        )

        if self.provider == "bocha":
            return self._search_bocha_docs(
                result=result,
                top_n_images=top_n_images,
                content_preview_len=content_preview_len,
                fetch_pages=fetch_pages,
            )

        return self._search_serpapi_docs(
            result=result,
            top_n_images=top_n_images,
            content_preview_len=content_preview_len,
            summary=summary,
            fetch_pages=fetch_pages,
        )
