# main.py
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from summarizer import summarize_spanish_article, create_deanna_ministore

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

app = FastAPI(title="Deanna Summarizer API")


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    topics: List[str]
    ministores: List[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Given article text, return two commercial topics and their ministore URLs.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Your function: returns list[str] of 2 topics
        topics = summarize_spanish_article(req.text)

        if not topics or len(topics) == 0:
            raise HTTPException(status_code=500, detail="No topics extracted")

        # For each topic, build ministore URL with your helper
        ministores = [create_deanna_ministore(t) for t in topics]

        return AnalyzeResponse(topics=topics, ministores=ministores)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# main.py
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from summarizer import summarize_spanish_article, create_deanna_ministore

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set")

app = FastAPI(title="Deanna Summarizer API")


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    topics: List[str]
    ministores: List[str]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Given article text, return two commercial topics and their ministore URLs.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # Your function: returns list[str] of 2 topics
        topics = summarize_spanish_article(req.text)

        if not topics or len(topics) == 0:
            raise HTTPException(status_code=500, detail="No topics extracted")

        # For each topic, build ministore URL with your helper
        ministores = [create_deanna_ministore(t) for t in topics]

        return AnalyzeResponse(topics=topics, ministores=ministores)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
