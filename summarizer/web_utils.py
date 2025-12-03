# web_utils.py
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup


def fetch_article_text_from_url(url: str, timeout: int = 10) -> str:
    """
    Fetches and roughly extracts main text content from a webpage.

    This is a simple heuristic approach (not a full article parser).
    For many news sites / blogs it will work decently.

    Parameters
    ----------
    url : str
        URL of the article.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Extracted visible text.
    """
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script / style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Try to focus on <article> if present
    article_tag = soup.find("article")
    if article_tag:
        text = article_tag.get_text(separator=" ", strip=True)
    else:
        # Fallback: use body text
        body = soup.body or soup
        text = body.get_text(separator=" ", strip=True)

    # Simple cleanup: collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text
