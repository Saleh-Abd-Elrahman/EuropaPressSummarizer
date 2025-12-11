# web_utils.py
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url


def _clean_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_generic_main_text(soup: BeautifulSoup) -> str:
    """
    Generic extractor for normal news/blog sites.
    Try article/content containers, then fallback to body text.
    """
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # 1) <article>
    article = soup.find("article")
    if article:
        t = article.get_text(" ", strip=True)
        if len(t.split()) > 30:
            return _clean_spaces(t)

    # 2) Common content divs
    for cls in [
        "entry-content",
        "post-content",
        "single-content",
        "td-post-content",
        "elementor-widget-theme-post-content",
        "content",
    ]:
        div = soup.find("div", class_=lambda c: c and cls in c)
        if div:
            t = div.get_text(" ", strip=True)
            if len(t.split()) > 30:
                return _clean_spaces(t)

    # 3) Fallback: whole body
    body = soup.body or soup
    t = body.get_text(" ", strip=True)
    return _clean_spaces(t)


def _extract_deanna_text(soup: BeautifulSoup) -> str:
    """
    Extremely simple extractor for deanna.today:
    just take ALL visible text in <body>.
    Noise is fine; GPT will summarise it.
    """
    body = soup.body or soup

    # Don't strip header/footer/nav here; keep everything to be safe
    for tag in body(["script", "style", "noscript"]):
        tag.decompose()

    text = body.get_text(" ", strip=True)
    return _clean_spaces(text)


def fetch_article_text_from_url(url: str, timeout: int = 10) -> str:
    """
    Fetch and extract article text from a URL.

    - For deanna.today: use very permissive extraction (whole body text).
    - For other sites: use a more targeted generic extractor.
    """
    url = _normalize_url(url)

    headers = DEFAULT_HEADERS.copy()
    if "deanna.today" in url:
        headers.update(
            {
                "Referer": "https://deanna.today/",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    resp = requests.get(url, headers=headers, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"El sitio respondió con código HTTP {resp.status_code}.")

    soup = BeautifulSoup(resp.text, "html.parser")

    if "deanna.today" in url:
        text = _extract_deanna_text(soup)
    else:
        text = _extract_generic_main_text(soup)

    if not text:
        raise RuntimeError(
            "Se obtuvo la página pero no se ha encontrado ningún texto."
        )

    # ⚠️ IMPORTANT: no minimum length check here for deanna.today.
    # Even if it’s noisy/short, let GPT handle it.
    return text
