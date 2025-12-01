import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("OpenAI.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIAdRecommender:
    def __init__(self, ads_df: pd.DataFrame):
        self.ads_df = ads_df.copy()
        self.ads_df["text"] = (
            self.ads_df["title"].astype(str) + " " +
            self.ads_df["description"].astype(str) + " " +
            self.ads_df.get("keywords", "").astype(str)
        )
        self.ads_df["embedding"] = self.ads_df["text"].apply(self._embed_text)

    def _embed_text(self, text: str) -> np.ndarray:
        r = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return np.array(r.data[0].embedding)

    def recommend_ads(self, article_text: str, top_k: int = 5) -> pd.DataFrame:
        article_emb = self._embed_text(article_text)
        sims = []
        for emb in self.ads_df["embedding"]:
            sims.append(cosine_similarity([article_emb], [emb])[0][0])
        self.ads_df["similarity"] = sims
        return (
            self.ads_df.sort_values("similarity", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
