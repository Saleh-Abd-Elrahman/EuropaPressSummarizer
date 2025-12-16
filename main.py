# main.py
import os
from typing import List

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from summarizer import (
    summarize_spanish_article,
    summarize_article_overall,
)

from ministore_creator import (
    get_db,
    create_ministore_in_db,
    render_ministore_html_from_db,
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

# ✅ IMPORTANT: this must be your public Render URL
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://europapresssummarizer.onrender.com").rstrip("/")

app = FastAPI(title="Deanna Summarizer API")


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeUrlRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    summary: str
    topics: List[str]
    ministores: List[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    paragraphs = [p.get_text(strip=True) for p in soup.select("article p") if p.get_text(strip=True)]
    if not paragraphs:
        paragraphs = [p.get_text(strip=True) for p in soup.select("p") if p.get_text(strip=True)]

    if paragraphs:
        text = "\n\n".join(paragraphs)
    else:
        text = soup.get_text(separator=" ", strip=True)

    text = " ".join(text.split())
    return text[:15000]


@app.get("/ministore/{ministore_id}", response_class=HTMLResponse)
async def ministore_page(ministore_id: str):
    """
    Public URL that the WP plugin will link to.
    Renders the ministore from DB.
    """
    db = get_db()
    try:
        html = render_ministore_html_from_db(db, ministore_id=ministore_id)
    finally:
        try:
            db.disconnect()
        except Exception:
            pass

    if not html:
        raise HTTPException(status_code=404, detail="Ministore not found")
    return HTMLResponse(content=html)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        summary = summarize_article_overall(req.text)
        topics = summarize_spanish_article(req.text)
        if not topics:
            raise HTTPException(status_code=500, detail="No topics extracted")

        # ✅ create real ministores in DB and return links to /ministore/<id>
        db = get_db()
        try:
            ministore_urls: List[str] = []
            for t in topics:
                created = create_ministore_in_db(
                    db=db,
                    topic=t,
                    language="es",
                    num_results=10,
                    items_to_link=8,
                )
                ministore_urls.append(f"{PUBLIC_BASE_URL}/ministore/{created.ministore_id}")
        finally:
            try:
                db.disconnect()
            except Exception:
                pass

        return AnalyzeResponse(summary=summary, topics=topics, ministores=ministore_urls)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_url", response_model=AnalyzeResponse)
async def analyze_url(req: AnalyzeUrlRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Empty URL")

    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; DeannaSummarizerBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error fetching URL: {e}")

    if resp.status_code != 200 or not resp.text:
        raise HTTPException(status_code=502, detail=f"Error fetching URL, HTTP {resp.status_code}")

    article_text = extract_text_from_html(resp.text)
    if not article_text:
        raise HTTPException(status_code=500, detail="No se ha podido extraer texto del artículo")

    try:
        summary = summarize_article_overall(article_text)
        topics = summarize_spanish_article(article_text)
        if not topics:
            raise HTTPException(status_code=500, detail="No topics extracted")

        db = get_db()
        try:
            ministore_urls: List[str] = []
            for t in topics:
                created = create_ministore_in_db(
                    db=db,
                    topic=t,
                    language="es",
                    num_results=10,
                    items_to_link=8,
                )
                ministore_urls.append(f"{PUBLIC_BASE_URL}/ministore/{created.ministore_id}")
        finally:
            try:
                db.disconnect()
            except Exception:
                pass

        return AnalyzeResponse(summary=summary, topics=topics, ministores=ministore_urls)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
