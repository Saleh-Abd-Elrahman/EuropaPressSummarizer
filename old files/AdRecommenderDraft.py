import os
import re
from typing import List, Set, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
from summarizer import summarize_spanish_article

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")
client = OpenAI(api_key=api_key)


def load_ads_from_txt(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError("Ads .txt file is empty.")
    blocks = re.split(r"\n\s*\n", content)
    records = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        title = ""
        description = ""
        keywords = ""
        for line in lines:
            lower = line.lower()
            if lower.startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif lower.startswith("description:"):
                description = line.split(":", 1)[1].strip()
            elif lower.startswith("keywords:"):
                keywords = line.split(":", 1)[1].strip()
        if title or description or keywords:
            records.append(
                {"title": title, "description": description, "keywords": keywords}
            )
    if not records:
        raise ValueError("No ads parsed from .txt file.")
    return pd.DataFrame.from_records(records)


class OpenAIAdRecommender:
    def __init__(self, ads_df: pd.DataFrame):
        self.ads_df = ads_df.copy()
        self.ads_df["title"] = self.ads_df["title"].astype(str)
        self.ads_df["description"] = self.ads_df["description"].astype(str)
        if "keywords" in self.ads_df.columns:
            self.ads_df["keywords"] = self.ads_df["keywords"].astype(str)
        else:
            self.ads_df["keywords"] = ""
        self.ads_df["text"] = (
            self.ads_df["title"]
            + " "
            + self.ads_df["description"]
            + " "
            + self.ads_df["keywords"]
        )
        self.ads_df["embedding"] = self.ads_df["text"].apply(self._embed_text)

    def _embed_text(self, text: str) -> np.ndarray:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _extract_keywords(self, text: str) -> Set[str]:
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        stopwords = {
            "la",
            "el",
            "los",
            "las",
            "un",
            "una",
            "unos",
            "unas",
            "y",
            "o",
            "a",
            "de",
            "del",
            "en",
            "por",
            "para",
            "con",
            "sin",
            "que",
            "es",
            "son",
            "se",
            "su",
            "sus",
            "al",
            "lo",
        }
        return {t for t in tokens if t not in stopwords and len(t) > 2}

    def recommend_ads(self, article_text: str, top_k: int = 5) -> pd.DataFrame:
        if not article_text or not article_text.strip():
            raise ValueError("Article text is empty.")
        article_text = article_text.strip()
        article_emb = self._embed_text(article_text)
        ad_embs = np.vstack(self.ads_df["embedding"].to_numpy())
        sims = cosine_similarity(article_emb.reshape(1, -1), ad_embs)[0]
        result = self.ads_df.copy()
        result["similarity"] = sims
        article_keywords = self._extract_keywords(article_text)
        ad_keywords_list: List[Set[str]] = [
            self._extract_keywords(text) for text in result["text"]
        ]
        matched_keywords_list = []
        keyword_overlap_list = []
        for ad_kw in ad_keywords_list:
            matched_kw = article_keywords.intersection(ad_kw)
            matched_keywords_list.append(", ".join(sorted(matched_kw)))
            keyword_overlap_list.append(len(matched_kw))
        result["matched_keywords"] = matched_keywords_list
        result["keyword_overlap"] = keyword_overlap_list
        return (
            result.sort_values(
                ["similarity", "keyword_overlap"],
                ascending=[False, False],
            )
            .head(top_k)
            .reset_index(drop=True)
        )

    def analyze_keywords(self, article_text: str, top_k: int = 5) -> pd.DataFrame:
        recommendations = self.recommend_ads(article_text=article_text, top_k=top_k)
        return recommendations[
            [
                "title",
                "description",
                "keywords",
                "similarity",
                "keyword_overlap",
                "matched_keywords",
            ]
        ]

    def analyze_article(
        self,
        raw_article_text: str,
        top_k: int = 5,
        summarize: bool = True,
        summary_max_chars: Optional[int] = 5000,
    ) -> pd.DataFrame:
        if not raw_article_text or not raw_article_text.strip():
            raise ValueError("Raw article text is empty.")
        raw_article_text = raw_article_text.strip()
        if summarize:
            summary = summarize_spanish_article(
                article_text=raw_article_text,
                max_chars=summary_max_chars,
            )
            text_for_matching = summary
        else:
            text_for_matching = raw_article_text
        return self.analyze_keywords(article_text=text_for_matching, top_k=top_k)


def recommend_ads_for_news_article(
    ads_txt_path: str,
    article_text: str,
    top_k: int = 5,
    summarize: bool = True,
    summary_max_chars: int = 2000,
) -> pd.DataFrame:
    ads_df = load_ads_from_txt(ads_txt_path)
    recommender = OpenAIAdRecommender(ads_df)
    return recommender.analyze_article(
        raw_article_text=article_text,
        top_k=top_k,
        summarize=summarize,
        summary_max_chars=summary_max_chars,
    )


if __name__ == "__main__":
    ads_txt_path = input("Path to ads .txt file: ").strip()
    article_text = input("Pega aquí el texto completo del artículo de noticia: ").strip()
    recommendations = recommend_ads_for_news_article(
        ads_txt_path=ads_txt_path,
        article_text=article_text,
        top_k=5,
        summarize=True,
        summary_max_chars=2000,
    )
    print(
        recommendations[
            [
                "title",
                "similarity",
                "matched_keywords",
            ]
        ]
    )
