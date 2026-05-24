import os
from typing import Any, Dict, List, Optional

import requests


BOCHA_API_URL = "https://api.bocha.cn/v1/web-search"


class WebSearcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = BOCHA_API_URL,
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key or os.getenv("BOCHA_API_KEY", "")
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()

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
        print(f"Search payload:\n{payload}\n")

        response = self.session.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        print(f"Search response status: {response.status_code}")

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

            grouped.setdefault(host_page_url, [])
            if image_url not in grouped[host_page_url]:
                grouped[host_page_url].append(image_url)

        return grouped

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

        print(f"Search engine web result count: {len(web_pages)}")
        print(f"Search engine image result count: {len(image_items)}")

        images_by_host = self._group_images_by_host(image_items)
        docs: List[Dict[str, Any]] = []

        for item in web_pages:
            url = str(item.get("url") or "").strip()
            if not url:
                continue

            summary_text = str(item.get("summary") or "").strip()
            snippet_text = str(item.get("snippet") or "").strip()
            content = summary_text or snippet_text

            image_urls = images_by_host.get(url, [])[:top_n_images]
            site_icon = str(item.get("siteIcon") or "").strip()
            if site_icon and site_icon not in image_urls and len(image_urls) < top_n_images:
                image_urls.append(site_icon)

            docs.append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "url": url,
                    "display_url": item.get("displayUrl"),
                    "content": content[:content_preview_len],
                    "full_content": content,
                    "snippet": snippet_text,
                    "summary": summary_text,
                    "site_name": item.get("siteName"),
                    "site_icon": item.get("siteIcon"),
                    "date_published": item.get("datePublished"),
                    "date_last_crawled": item.get("dateLastCrawled"),
                    "cached_page_url": item.get("cachedPageUrl"),
                    "language": item.get("language"),
                    "is_family_friendly": item.get("isFamilyFriendly"),
                    "is_navigational": item.get("isNavigational"),
                    "image_urls": image_urls,
                }
            )

        print(f"\nurls:\n{[doc['url'] for doc in docs]}\n")
        return docs


def main() -> None:
    web = WebSearcher(api_key="sk-XXX")
    query = "Where is Taihang mountain?"

    print(f"Query: {query}")

    docs = web.search(
        query=query,
        candidate_k=10,
        summary=True,
        freshness="noLimit", 
        top_n_images=3,
        content_preview_len=800,
    )

    print(f"Retrieved docs: {len(docs)}")


if __name__ == "__main__":
    main()
