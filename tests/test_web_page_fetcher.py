import shutil
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from rag_v1.config import PROJECT_ROOT
from rag_v1.services.web_page_fetcher import (
    FetchedPage,
    WebPageFetcher,
    canonicalize_url,
    extract_html_document,
    is_valid_page_image_url,
)
from rag_v1.services.web_search import WebSearcher


ARTICLE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <title>Article Title - Example News</title>
    <meta property="og:title" content="Article Title">
    <meta property="og:site_name" content="Example News">
    <meta property="article:published_time" content="2026-05-04T10:00:00Z">
    <meta property="og:image" content="/images/hero.jpg">
  </head>
  <body>
    <nav>Home | Login | Subscribe | Share this</nav>
    <main class="article-content">
      <h1>Article Title</h1>
      <p>The first paragraph contains the core evidence and should be retained by the extractor.</p>
      <p>The second paragraph adds more specific facts, context, and enough length for RAG chunking.</p>
      <p>The second paragraph adds more specific facts, context, and enough length for RAG chunking.</p>
    </main>
    <aside>Related articles and recommendations</aside>
    <footer>Copyright 2026 Example News. All rights reserved.</footer>
  </body>
</html>
"""


class WebPageFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = PROJECT_ROOT / ".test_tmp" / "web_page_fetcher_tests"
        self._assert_under_project(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._assert_under_project(self.test_root)
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    @staticmethod
    def _assert_under_project(path: Path) -> None:
        resolved_path = path.resolve()
        project_root = PROJECT_ROOT.resolve()
        resolved_path.relative_to(project_root)

    def test_canonicalize_url_drops_tracking_and_fragment(self) -> None:
        url = "HTTPS://Example.com:443/a/b/?utm_source=x&b=2&a=1#section"
        self.assertEqual(canonicalize_url(url), "https://example.com/a/b?a=1&b=2")
        self.assertEqual(canonicalize_url("https://example.com:bad/path"), "")

    def test_image_url_filter_rejects_non_images_and_tiny_assets(self) -> None:
        self.assertTrue(is_valid_page_image_url("https://example.com/news/photo.jpg"))
        self.assertTrue(is_valid_page_image_url("https://cdn.example.com/image?id=12345"))
        self.assertFalse(is_valid_page_image_url("https://example.com/static/app.js"))
        self.assertFalse(is_valid_page_image_url("https://example.com/static/site.css"))
        self.assertFalse(is_valid_page_image_url("https://example.com/fonts/news.woff2"))
        self.assertFalse(is_valid_page_image_url("https://example.com/favicon.ico"))
        self.assertFalse(is_valid_page_image_url("https://example.com/images/logo.svg"))
        self.assertFalse(is_valid_page_image_url("https://example.com/images/bgr_navleft.jpg"))
        self.assertFalse(is_valid_page_image_url("https://example.com/images/channel_tag_3.jpg"))

    def test_extract_article_text_and_metadata(self) -> None:
        page = extract_html_document(
            requested_url="https://example.com/story?utm_source=x",
            final_url="https://example.com/story",
            html=ARTICLE_HTML,
            status_code=200,
        )

        self.assertTrue(page.ok)
        self.assertEqual(page.title, "Article Title")
        self.assertEqual(page.site_name, "Example News")
        self.assertEqual(page.date_published, "2026-05-04T10:00:00Z")
        self.assertIn("core evidence", page.text)
        self.assertNotIn("Home | Login", page.text)
        self.assertNotIn("Copyright 2026", page.text)
        self.assertEqual(page.text.count("second paragraph adds"), 1)
        self.assertEqual(page.image_urls[0], "https://example.com/images/hero.jpg")

    def test_fetcher_uses_cache_on_second_request(self) -> None:
        class CountingHandler(BaseHTTPRequestHandler):
            request_count = 0

            def do_GET(self) -> None:
                type(self).request_count += 1
                body = ARTICLE_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), CountingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/story?utm_source=x"
            fetcher = WebPageFetcher(
                cache_dir=self.test_root / "cache",
                max_retries=0,
                cache_ttl_seconds=3600,
                min_text_chars=20,
            )
            first = fetcher.fetch(url)
            second = fetcher.fetch(url)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertTrue(first.ok)
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(CountingHandler.request_count, 1)
        self.assertEqual(first.content_hash, second.content_hash)

    def test_web_search_uses_fetched_body_before_bocha_summary(self) -> None:
        full_text = (
            "This is a fetched webpage body with real article evidence. "
            "It is intentionally longer than the search summary so the RAG "
            "pipeline can prefer the page body over the snippet. "
        ) * 4

        class FakeFetcher:
            def fetch(self, url: object) -> FetchedPage:
                return FetchedPage(
                    requested_url=str(url),
                    canonical_url="https://example.com/story",
                    final_url="https://example.com/story",
                    status_code=200,
                    ok=True,
                    fetch_status="success",
                    title="Fetched Article",
                    text=full_text,
                    site_name="Example News",
                    image_urls=["https://example.com/hero.jpg"],
                    content_hash="abc123",
                    extraction_method="test",
                    quality_score=0.8,
                )

        searcher = WebSearcher(api_key="test")
        searcher.page_fetcher = FakeFetcher()
        searcher.search_engine = lambda **kwargs: {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "id": "1",
                            "name": "Bocha Title",
                            "url": "https://example.com/story?utm_source=x",
                            "summary": "Short Bocha summary.",
                            "snippet": "Short snippet.",
                            "siteName": "",
                        }
                    ]
                },
                "images": {
                    "value": [
                        {
                            "hostPageUrl": "https://example.com/story?utm_source=x",
                            "contentUrl": "https://example.com/static/app.js",
                        },
                        {
                            "hostPageUrl": "https://example.com/story?utm_source=x",
                            "contentUrl": "https://example.com/gallery/photo.jpg",
                        },
                    ]
                },
            }
        }

        docs = searcher.search("query", candidate_k=1, fetch_pages=True)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content_source"], "web_page")
        self.assertEqual(docs[0]["name"], "Fetched Article")
        self.assertIn("real article evidence", docs[0]["full_content"])
        self.assertEqual(docs[0]["summary"], "Short Bocha summary.")
        self.assertEqual(docs[0]["web_fetch_quality_score"], 0.8)
        self.assertEqual(
            docs[0]["image_urls"],
            ["https://example.com/gallery/photo.jpg", "https://example.com/hero.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
