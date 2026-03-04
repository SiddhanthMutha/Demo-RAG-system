"""
Web parser: scrapes HTML from URLs using requests + BeautifulSoup4.
Respects rate limiting and extracts clean text from main content.
"""
import re
import time
from typing import Any, Dict, List
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.ingestion.base_parser import BaseParser, ParsingError
from src.models import Document, DocumentType

_RATE_LIMIT_DELAY = 1.0  # seconds between requests
_last_request_time: float = 0.0

USER_AGENT = "RAGSystemBot/1.0"
HEADERS = {"User-Agent": USER_AGENT}

# Tags that contain main content (ordered by priority)
CONTENT_TAGS = ["article", "main", "section", "div[role='main']", ".content", "#content"]

# Tags to strip (navigation, ads, etc.)
STRIP_TAGS = [
    "nav", "header", "footer", "aside", "script", "style",
    "noscript", "iframe", "form", "button", "advertisement",
]


class WebParser(BaseParser):
    """Parses web pages (HTML) into Document objects."""

    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.WEB]

    async def parse(self, source: str, **kwargs: Any) -> Document:
        """
        Fetch and parse a web page.

        Args:
            source: URL to scrape.

        Returns:
            Document with extracted markdown-like text and metadata.

        Raises:
            ParsingError: If the URL can't be fetched or has no content.
        """
        url = source.strip()
        if not url.startswith(("http://", "https://")):
            raise ParsingError(f"Invalid URL scheme: {url}", source=url)

        self._respect_rate_limit()
        self._check_robots_txt(url)

        logger.info("Fetching web page", url=url)

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=15,
                allow_redirects=True,
                max_redirects=3,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise ParsingError(f"HTTP request failed: {e}", source=url, cause=e)

        soup = BeautifulSoup(response.text, "lxml")
        text, metadata = self._extract_content(soup, url)

        if not text.strip():
            raise ParsingError(f"No content extracted from {url}", source=url)

        return Document(
            content=text,
            doc_type=DocumentType.WEB,
            source=url,
            metadata={**metadata, "content_type": response.headers.get("content-type", "")},
        )

    def _extract_content(self, soup: BeautifulSoup, url: str) -> tuple[str, Dict[str, Any]]:
        """Extract main text and metadata from parsed HTML."""
        # Remove noise elements
        for tag in soup.find_all(STRIP_TAGS):
            tag.decompose()

        # Extract metadata from <head>
        metadata: Dict[str, Any] = {
            "title": self._get_title(soup),
            "author": self._get_meta(soup, ["author", "article:author"]),
            "description": self._get_meta(soup, ["description", "og:description"]),
            "publish_date": self._get_meta(soup, ["article:published_time", "datePublished"]),
            "url": url,
        }

        # Try to find the main content container
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", role="main")
            or soup.find("div", id="content")
            or soup.find("div", class_="content")
            or soup.body
        )

        if main_content is None:
            main_content = soup

        # Extract text, preserving some structure
        text_parts: List[str] = []
        for elem in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"]):
            text = elem.get_text(separator=" ", strip=True)
            if text:
                if elem.name and elem.name.startswith("h"):
                    text_parts.append(f"\n## {text}\n")
                else:
                    text_parts.append(text)

        return "\n".join(text_parts), metadata

    def _get_title(self, soup: BeautifulSoup) -> str:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return str(og_title["content"])
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""

    def _get_meta(self, soup: BeautifulSoup, names: List[str]) -> str:
        for name in names:
            tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", property=name)
            if tag and tag.get("content"):
                return str(tag["content"])
        return ""

    def _respect_rate_limit(self) -> None:
        """Enforce 1 request/second rate limiting."""
        global _last_request_time
        elapsed = time.time() - _last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            time.sleep(_RATE_LIMIT_DELAY - elapsed)
        _last_request_time = time.time()

    def _check_robots_txt(self, url: str) -> None:
        """Check robots.txt and warn if crawling is disallowed (non-blocking)."""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            if not rp.can_fetch(USER_AGENT, url):
                logger.warning("robots.txt disallows crawling", url=url)
        except Exception:
            pass  # Don't block on robots.txt failures
