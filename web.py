import os
from typing import Any, Dict, Optional

import requests


def _serpapi_tbs_from_freshness(freshness: str) -> Optional[str]:
    mapping = {
        "oneDay": "qdr:d",
        "oneWeek": "qdr:w",
        "oneMonth": "qdr:m",
        "oneYear": "qdr:y",
    }
    return mapping.get(str(freshness or "").strip())


class WebSearcher:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 20,
        provider: Optional[str] = None,
    ) -> None:
        self.provider = (provider or os.getenv("WEB_SEARCH_PROVIDER", "serpapi")).strip().lower()
        self.timeout = timeout
        self.session = requests.Session()

        if self.provider == "bocha":
            self.api_key = api_key or os.getenv("BOCHA_API_KEY", "")
            self.api_url = api_url or os.getenv("BOCHA_API_URL", "https://api.bochaai.com/v1/web-search")
        elif self.provider == "serpapi":
            self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")
            self.api_url = api_url or os.getenv("SERPAPI_API_URL", "https://serpapi.com/search.json")
        else:
            raise ValueError("WEB_SEARCH_PROVIDER must be 'bocha' or 'serpapi'.")

        if not self.api_key:
            raise ValueError(
                f"Search API key is required for provider={self.provider!r}."
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

        print(f"\nSearch provider:\n{self.provider}\n")
        print(f"Search url:\n{self.api_url}\n")

        if self.provider == "bocha":
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
        else:
            if count < 1 or count > 100:
                raise ValueError("count must be between 1 and 100")
            effective_query = query.strip()
            if include:
                effective_query = f"{effective_query} {include}".strip()
            if exclude:
                effective_query = f"{effective_query} {exclude}".strip()
            params: Dict[str, Any] = {
                "engine": os.getenv("SERPAPI_ENGINE", "google"),
                "q": effective_query,
                "api_key": self.api_key,
                "google_domain": os.getenv("SERPAPI_GOOGLE_DOMAIN", "google.com"),
                "gl": os.getenv("SERPAPI_GL", "us"),
                "hl": os.getenv("SERPAPI_HL", "en"),
                "num": count,
                "output": "json",
            }
            location = os.getenv("SERPAPI_LOCATION", "").strip()
            if location:
                params["location"] = location
            if os.getenv("SERPAPI_NO_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}:
                params["no_cache"] = "true"
            tbs = _serpapi_tbs_from_freshness(freshness)
            if tbs:
                params["tbs"] = tbs
            print(f"Search params:\n{params}\n")

            response = self.session.get(
                self.api_url,
                params=params,
                timeout=self.timeout,
            )

        print(f"Search response status: {response.status_code}")

        try:
            result = response.json()
        except ValueError as exc:
            response.raise_for_status()
            raise RuntimeError("Search API did not return valid JSON") from exc

        return result
