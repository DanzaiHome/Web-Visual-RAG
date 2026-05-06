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

    def test_article_body_blocks_survive_high_score_related_titles(self) -> None:
        html = """
        <html>
          <head>
            <meta property="og:title" content="Retirement Access Fact Sheet">
            <meta property="article:published_time" content="2026-05-04T10:00:00Z">
          </head>
          <body>
            <main class="article-content">
              <h1>Retirement Access Fact Sheet</h1>
              <p>DELIVERING RETIREMENT SECURITY TO WORKING AMERICANS: The order expands access to low-cost retirement accounts for workers who do not have employer-sponsored plans.</p>
              <ul>
                <li>The Order directs the Treasury Secretary to establish a federal platform that connects workers with private-sector Individual Retirement Accounts.</li>
                <li>Eligible lower-income workers can receive federal matching contributions designed to help build retirement savings.</li>
                <li>The platform allows workers to compare retirement accounts by cost, quality, and investment options so they can choose a suitable savings path.</li>
                <li>The policy also asks agencies to prepare guidance and legislative recommendations that preserve access to low-cost savings vehicles.</li>
                <li>Independent contractors, part-time employees, small business workers, and self-employed individuals are among the groups expected to benefit.</li>
              </ul>
              <h4>Fact Sheet: Unrelated Fertility Care Announcement</h4>
              <h4>Fact Sheet: Unrelated Alternative Assets Announcement With a Longer Headline That Should Still Be Treated as a Related Link</h4>
            </main>
          </body>
        </html>
        """

        page = extract_html_document(
            requested_url="https://example.com/fact-sheet",
            final_url="https://example.com/fact-sheet",
            html=html,
            status_code=200,
        )

        self.assertFalse(page.is_probable_listing)
        self.assertGreater(page.quality_score, 0.35)
        self.assertIn("Treasury Secretary", page.text)
        self.assertNotIn("Unrelated Fertility", page.text)
        self.assertNotIn("Alternative Assets Announcement", page.text)

    def test_extract_canonical_link_and_filters_page_images(self) -> None:
        html = """
        <html>
          <head>
            <link rel="canonical" href="https://example.com/article?id=42&utm_source=x#comments">
            <link rel="preload" href="/static/app.css" as="style">
            <link rel="preload" href="/images/preloaded.jpg" as="image">
            <meta property="og:image" content="/images/social.webp">
          </head>
          <body>
            <main class="article-content">
              <h1>Canonical Article</h1>
              <p>This article contains enough clean evidence to validate canonical extraction.</p>
              <img src="/tracking/pixel.png" width="1" height="1">
              <img src="/images/logo.png" width="600" height="300">
              <img src="/images/brand.png" class="site-logo" width="600" height="300">
              <img src="/images/bgr_navleft.jpg" width="640" height="360">
              <img src="/images/body-photo.png" width="640" height="360">
              <img src="/assets/main.js" width="640" height="360">
            </main>
          </body>
        </html>
        """

        page = extract_html_document(
            requested_url="https://example.com/article?id=42&utm_source=x",
            final_url="https://example.com/article?id=42&utm_source=x",
            html=html,
            status_code=200,
        )

        self.assertEqual(page.canonical_url, "https://example.com/article?id=42")
        self.assertIn("https://example.com/images/preloaded.jpg", page.image_urls)
        self.assertIn("https://example.com/images/social.webp", page.image_urls)
        self.assertIn("https://example.com/images/body-photo.png", page.image_urls)
        self.assertNotIn("https://example.com/static/app.css", page.image_urls)
        self.assertNotIn("https://example.com/tracking/pixel.png", page.image_urls)
        self.assertNotIn("https://example.com/images/logo.png", page.image_urls)
        self.assertNotIn("https://example.com/images/brand.png", page.image_urls)
        self.assertNotIn("https://example.com/images/bgr_navleft.jpg", page.image_urls)
        self.assertNotIn("https://example.com/assets/main.js", page.image_urls)

    def test_article_header_title_is_kept_but_site_header_is_dropped(self) -> None:
        html = """
        <html>
          <body>
            <header>
              <h1>Site Header</h1>
              <p>Navigation text should not appear.</p>
              <img src="/images/site-brand.png" width="500" height="200">
            </header>
            <article>
              <header><h1>Article Header Title</h1></header>
              <p>The article body remains available as clean evidence for retrieval.</p>
              <img src="/images/article-photo.jpg" width="500" height="280">
            </article>
          </body>
        </html>
        """

        page = extract_html_document(
            requested_url="https://example.com/article",
            final_url="https://example.com/article",
            html=html,
            status_code=200,
        )

        self.assertEqual(page.title, "Article Header Title")
        self.assertIn("Article Header Title", page.text)
        self.assertIn("clean evidence", page.text)
        self.assertNotIn("Site Header", page.text)
        self.assertNotIn("Navigation text", page.text)
        self.assertIn("https://example.com/images/article-photo.jpg", page.image_urls)
        self.assertNotIn("https://example.com/images/site-brand.png", page.image_urls)

    def test_title_listing_page_gets_low_quality_score(self) -> None:
        related_items = "\n".join(
            f"<li>Global policy headline number {index} with extra words</li>"
            for index in range(30)
        )
        html = f"""
        <html>
          <head>
            <meta property="og:title" content="Policy Story">
            <meta property="article:published_time" content="2026-05-04T10:00:00Z">
          </head>
          <body>
            <main class="article-content">
              <h1>Policy Story</h1>
              <ul>
                {related_items}
              </ul>
            </main>
          </body>
        </html>
        """

        page = extract_html_document(
            requested_url="https://example.com/listing",
            final_url="https://example.com/listing",
            html=html,
            status_code=200,
        )

        self.assertTrue(page.is_probable_listing)
        self.assertLess(page.quality_score, 0.25)
        self.assertIn("Policy Story", page.text)

    def test_short_title_list_page_gets_listing_flag(self) -> None:
        html = """
        <html>
          <head>
            <meta property="og:title" content="Retirement Access Fact Sheet">
            <meta property="article:published_time" content="2026-05-04T10:00:00Z">
          </head>
          <body>
            <main class="article-content">
              <h1>Retirement Access Fact Sheet</h1>
              <h2>Promoting Retirement Savings Access for American Workers</h2>
              <h2>Presidential Actions and Executive Orders</h2>
              <h2>Fact Sheet: Alternative Assets for Retirement Investors</h2>
              <h2>Fact Sheet: Expanding Access to Fertility Care</h2>
              <h2>President Is Delivering for American Workers</h2>
            </main>
          </body>
        </html>
        """

        page = extract_html_document(
            requested_url="https://example.com/title-list",
            final_url="https://example.com/title-list",
            html=html,
            status_code=200,
        )

        self.assertTrue(page.is_probable_listing)
        self.assertLess(page.quality_score, 0.2)

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

    def test_fetcher_uses_meta_charset_when_header_omits_charset(self) -> None:
        gbk_html = """
        <html>
          <head><meta charset="gbk"><title>编码测试</title></head>
          <body>
            <main class="article-content">
              <p>这是一段中文正文，用来验证网页头部缺少 charset 时仍然可以按照 meta 编码解码。</p>
              <p>第二段继续提供足够的正文长度，避免被抽取器误判为空页面。</p>
            </main>
          </body>
        </html>
        """

        class GbkHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = gbk_html.encode("gbk")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), GbkHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/gbk"
            fetcher = WebPageFetcher(
                cache_dir=self.test_root / "gbk-cache",
                max_retries=0,
                cache_ttl_seconds=3600,
                min_text_chars=20,
            )
            page = fetcher.fetch(url)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertTrue(page.ok)
        self.assertIn("中文正文", page.text)
        self.assertNotIn("\ufffd", page.text)

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

    def test_web_search_prefers_summary_over_low_quality_listing_fetch(self) -> None:
        listing_text = "\n\n".join(
            f"Related headline {index} with shallow navigation text"
            for index in range(20)
        )

        class ListingFetcher:
            def fetch(self, url: object) -> FetchedPage:
                return FetchedPage(
                    requested_url=str(url),
                    canonical_url="https://example.com/listing",
                    final_url="https://example.com/listing",
                    status_code=200,
                    ok=True,
                    fetch_status="success",
                    title="Fetched Listing",
                    text=listing_text,
                    image_urls=[],
                    content_hash="listing123",
                    extraction_method="test",
                    quality_score=0.05,
                    is_probable_listing=True,
                )

        searcher = WebSearcher(api_key="test")
        searcher.page_fetcher = ListingFetcher()
        searcher.search_engine = lambda **kwargs: {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "id": "1",
                            "name": "Bocha Title",
                            "url": "https://example.com/listing",
                            "summary": "Clean Bocha summary with useful policy facts. " * 10,
                            "snippet": "Short snippet.",
                        }
                    ]
                },
                "images": {"value": []},
            }
        }

        docs = searcher.search("query", candidate_k=1, fetch_pages=True)

        self.assertEqual(docs[0]["content_source"], "bocha_summary_fallback")
        self.assertIn("Clean Bocha summary", docs[0]["full_content"])
        self.assertEqual(docs[0]["web_fetch_status"], "success")
        self.assertEqual(docs[0]["web_fetch_quality_score"], 0.05)

    def test_web_search_falls_back_to_bocha_summary_when_fetch_fails(self) -> None:
        class FailingFetcher:
            def fetch(self, url: object) -> FetchedPage:
                return FetchedPage(
                    requested_url=str(url),
                    canonical_url="https://example.com/missing",
                    final_url="https://example.com/missing",
                    status_code=404,
                    ok=False,
                    fetch_status="http_error",
                    error="HTTP 404",
                )

        searcher = WebSearcher(api_key="test")
        searcher.page_fetcher = FailingFetcher()
        searcher.search_engine = lambda **kwargs: {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "id": "1",
                            "name": "Bocha Title",
                            "url": "https://example.com/missing",
                            "summary": "Bocha summary remains available.",
                            "snippet": "Short snippet.",
                        }
                    ]
                },
                "images": {"value": []},
            }
        }

        docs = searcher.search("query", candidate_k=1, fetch_pages=True)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content_source"], "bocha_summary_fallback")
        self.assertEqual(docs[0]["web_fetch_status"], "http_error")
        self.assertEqual(docs[0]["full_content"], "Bocha summary remains available.")


if __name__ == "__main__":
    unittest.main()
