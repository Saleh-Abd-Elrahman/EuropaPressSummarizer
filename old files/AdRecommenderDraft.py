import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Set, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI

from summarizer import summarize_spanish_article
from MySQLConnector import MySQLConnector

load_dotenv()
load_dotenv("SerperKey.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")

SERPER_API_KEY = os.getenv("Serper.dev_Key")

client = OpenAI(api_key=OPENAI_API_KEY)


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


def _call_serper(query: str, num_results: int = 10, lang: str = "es") -> Dict:
    if not SERPER_API_KEY:
        raise RuntimeError("Serper.dev_Key not found in environment.")

    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num_results, "hl": lang}
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp_data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Serper HTTP error: {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Serper connection error: {e.reason}") from e

    try:
        return json.loads(resp_data)
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to decode Serper response as JSON.") from e


def fetch_ministore_items_from_serper(
    query: str,
    num_results: int = 10,
    language: str = "es",
) -> pd.DataFrame:
    data = _call_serper(query=query, num_results=num_results, lang=language)

    items = data.get("shopping") or data.get("organic") or []
    records: List[Dict[str, str]] = []

    for idx, item in enumerate(items):
        title = item.get("title") or ""
        description = (
            item.get("snippet")
            or item.get("description")
            or ""
        )
        url = item.get("link") or item.get("productLink") or ""
        if not title and not url:
            continue

        records.append(
            {
                "id": str(item.get("productId", idx)),
                "title": title,
                "description": description,
                "url": url,
                "keywords": query,
                "language": language,
            }
        )

    if not records:
        raise ValueError("Serper returned no usable results for ministore items.")
    return pd.DataFrame.from_records(records)


def load_ministore_items_from_db(
    db: MySQLConnector,
    table_name: str = "ministore_items",
) -> pd.DataFrame:
    sql = f"""
        SELECT
            id,
            title,
            description,
            url,
            keywords,
            language
        FROM {table_name}
    """
    rows = db.execute_query(sql)
    if rows is None:
        raise RuntimeError("Failed to load ministore items from database.")
    if not rows:
        raise ValueError("No ministore items found in database.")
    return pd.DataFrame(rows)


def upsert_ministore_items_into_db(
    db: MySQLConnector,
    items_df: pd.DataFrame,
    table_name: str = "ministore_items",
) -> int:
    if items_df.empty:
        return 0

    sql = f"""
        INSERT INTO {table_name} (id, title, description, url, keywords, language)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            description = VALUES(description),
            url = VALUES(url),
            keywords = VALUES(keywords),
            language = VALUES(language)
    """

    values = [
        (
            str(row["id"]),
            str(row["title"]),
            str(row["description"]),
            str(row["url"]),
            str(row["keywords"]),
            str(row["language"]),
        )
        for _, row in items_df.iterrows()
    ]

    if not db.connection or not db.connection.is_connected():
        raise RuntimeError("MySQLConnector is not connected. Call connect() first.")

    cursor = None
    try:
        cursor = db.connection.cursor()
        cursor.executemany(sql, values)
        db.connection.commit()
        return cursor.rowcount
    finally:
        if cursor:
            cursor.close()


def load_user_interactions_from_db(
    db: MySQLConnector,
    table_name: str = "ministore_interactions",
) -> pd.DataFrame:
    sql = f"""
        SELECT
            user_id,
            item_id,
            interaction_type,
            dwell_time
        FROM {table_name}
    """
    rows = db.execute_query(sql)
    if rows is None:
        raise RuntimeError("Failed to load user interactions from database.")
    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "interaction_type", "dwell_time"])
    return pd.DataFrame(rows)


class OpenAIAdRecommender:
    def __init__(self, ads_df: pd.DataFrame):
        self.ads_df = ads_df.copy()

        self.ads_df["title"] = self.ads_df["title"].astype(str)
        self.ads_df["description"] = self.ads_df["description"].astype(str)

        if "keywords" in self.ads_df.columns:
            self.ads_df["keywords"] = self.ads_df["keywords"].astype(str)
        else:
            self.ads_df["keywords"] = ""

        if "url" not in self.ads_df.columns:
            self.ads_df["url"] = ""
        else:
            self.ads_df["url"] = self.ads_df["url"].astype(str)

        if "language" not in self.ads_df.columns:
            self.ads_df["language"] = ""
        else:
            self.ads_df["language"] = self.ads_df["language"].astype(str)

        if "id" in self.ads_df.columns:
            self.ads_df["item_id"] = self.ads_df["id"].astype(str)
        else:
            self.ads_df["item_id"] = self.ads_df.index.astype(str)

        self.ads_df["text"] = (
            self.ads_df["title"]
            + " "
            + self.ads_df["description"]
            + " "
            + self.ads_df["keywords"]
        )

        self._embedding_cache: Dict[str, np.ndarray] = {}

        self.ads_df["embedding"] = self.ads_df["text"].apply(self._embed_text)

        self._item_id_to_idx: Dict[str, int] = {
            item_id: idx for idx, item_id in enumerate(self.ads_df["item_id"])
        }
        self._item_embeddings_matrix: np.ndarray = np.vstack(
            self.ads_df["embedding"].to_numpy()
        )

        self._user_profiles: Dict[str, np.ndarray] = {}
        self._user_item_weights: Dict[str, Dict[str, float]] = {}
        self._user_interaction_counts: Dict[str, int] = {}
        self.min_user_interactions: int = 3

    def _embed_text(self, text: str) -> np.ndarray:
        key = text.strip()
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=key,
        )
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self._embedding_cache[key] = emb
        return emb

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

        ad_embs = self._item_embeddings_matrix
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

    def recommend_similar_items(
        self,
        item_id: str,
        top_n: int = 5,
        include_item_itself: bool = False,
    ) -> pd.DataFrame:
        if item_id not in self._item_id_to_idx:
            raise KeyError(f"Unknown item_id: {item_id}")

        anchor_idx = self._item_id_to_idx[item_id]
        anchor_emb = self._item_embeddings_matrix[anchor_idx].reshape(1, -1)
        sims = cosine_similarity(anchor_emb, self._item_embeddings_matrix)[0]

        result = self.ads_df.copy()
        result["similarity_item"] = sims

        if not include_item_itself:
            result = result[result["item_id"] != item_id]

        return (
            result.sort_values("similarity_item", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def fit_user_interactions(self, interactions: pd.DataFrame) -> None:
        if interactions.empty:
            self._user_profiles.clear()
            self._user_item_weights.clear()
            self._user_interaction_counts.clear()
            return

        interactions = interactions.copy()
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["item_id"] = interactions["item_id"].astype(str)

        required_cols = {"user_id", "item_id"}
        missing = required_cols - set(interactions.columns)
        if missing:
            raise ValueError(f"Missing required interaction columns: {missing}")

        type_weights = {
            "view": 1.0,
            "click": 2.0,
            "like": 3.0,
            "purchase": 4.0,
        }

        self._user_profiles = {}
        self._user_item_weights = {}
        self._user_interaction_counts = {}

        for _, row in interactions.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]

            if item_id not in self._item_id_to_idx:
                continue

            interaction_type = str(row.get("interaction_type", "") or "").lower()
            dwell_time = row.get("dwell_time", None)

            base_weight = type_weights.get(interaction_type, 1.0)

            if dwell_time is not None:
                try:
                    dt = float(dwell_time)
                    if dt > 0:
                        dwell_factor = min(dt / 10.0, 3.0)
                        weight = base_weight * dwell_factor
                    else:
                        weight = base_weight
                except (TypeError, ValueError):
                    weight = base_weight
            else:
                weight = base_weight

            user_items = self._user_item_weights.setdefault(user_id, {})
            user_items[item_id] = user_items.get(item_id, 0.0) + weight
            self._user_interaction_counts[user_id] = (
                self._user_interaction_counts.get(user_id, 0) + 1
            )

        for user_id, item_weights in self._user_item_weights.items():
            if not item_weights:
                continue
            total_weight = sum(item_weights.values())
            if total_weight <= 0:
                continue

            dim = self._item_embeddings_matrix.shape[1]
            profile_vec = np.zeros(dim, dtype=np.float32)

            for item_id, w in item_weights.items():
                idx = self._item_id_to_idx[item_id]
                profile_vec += w * self._item_embeddings_matrix[idx]

            profile_vec /= total_weight
            self._user_profiles[user_id] = profile_vec

    def _user_similarity(self, target_user_id: str) -> Dict[str, float]:
        if target_user_id not in self._user_profiles:
            return {}

        target_vec = self._user_profiles[target_user_id].reshape(1, -1)
        other_ids: List[str] = [
            uid for uid in self._user_profiles.keys() if uid != target_user_id
        ]
        if not other_ids:
            return {}

        other_vecs = np.vstack([self._user_profiles[uid] for uid in other_ids])
        sims = cosine_similarity(target_vec, other_vecs)[0]
        return {uid: float(sim) for uid, sim in zip(other_ids, sims)}

    def recommend_items_user_based(
        self,
        user_id: str,
        top_n: int = 5,
        similar_users_k: int = 20,
    ) -> pd.DataFrame:
        if user_id not in self._user_profiles:
            raise KeyError(f"Unknown user_id: {user_id}")

        user_sim = self._user_similarity(user_id)
        if not user_sim:
            return self.ads_df.head(0)

        neighbors = sorted(
            user_sim.items(), key=lambda x: x[1], reverse=True
        )[:similar_users_k]

        target_seen = set(self._user_item_weights.get(user_id, {}).keys())
        item_scores: Dict[str, float] = {}

        for neighbor_id, sim in neighbors:
            if sim <= 0:
                continue
            neighbor_items = self._user_item_weights.get(neighbor_id, {})
            for item_id, w in neighbor_items.items():
                if item_id in target_seen:
                    continue
                item_scores[item_id] = item_scores.get(item_id, 0.0) + sim * w

        if not item_scores:
            return self.ads_df.head(0)

        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        ranked_ids = [item_id for item_id, _ in ranked]

        result = self.ads_df.set_index("item_id").loc[ranked_ids].copy()
        result["score_user_based"] = [score for _, score in ranked]
        return result.reset_index()

    def recommend_items_hybrid(
        self,
        user_id: str,
        anchor_item_id: str,
        top_n: int = 5,
        alpha: float = 0.6,
    ) -> pd.DataFrame:
        if anchor_item_id not in self._item_id_to_idx:
            raise KeyError(f"Unknown anchor_item_id: {anchor_item_id}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0.0 and 1.0")

        item_based_df = self.recommend_similar_items(
            item_id=anchor_item_id,
            top_n=len(self.ads_df),
            include_item_itself=False,
        )
        item_scores = dict(
            zip(item_based_df["item_id"], item_based_df["similarity_item"])
        )

        user_interactions = self._user_interaction_counts.get(user_id, 0)
        if user_interactions < self.min_user_interactions or user_id not in self._user_profiles:
            return item_based_df.head(top_n).reset_index(drop=True)

        user_based_df = self.recommend_items_user_based(
            user_id=user_id,
            top_n=len(self.ads_df),
        )
        user_scores = dict(
            zip(user_based_df["item_id"], user_based_df["score_user_based"])
        )

        def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            max_val = max(scores.values())
            if max_val <= 0:
                return {k: 0.0 for k in scores}
            return {k: v / max_val for k, v in scores.items()}

        item_scores_norm = _normalize(item_scores)
        user_scores_norm = _normalize(user_scores)

        excluded_items = set(self._user_item_weights.get(user_id, {}).keys())
        excluded_items.add(anchor_item_id)

        combined: Dict[str, float] = {}
        candidate_ids = set(item_scores_norm.keys()) | set(user_scores_norm.keys())

        for item_id in candidate_ids:
            if item_id in excluded_items:
                continue
            s_item = item_scores_norm.get(item_id, 0.0)
            s_user = user_scores_norm.get(item_id, 0.0)
            combined[item_id] = alpha * s_item + (1.0 - alpha) * s_user

        if not combined:
            return item_based_df.head(top_n).reset_index(drop=True)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
        ranked_ids = [item_id for item_id, _ in ranked]

        result = self.ads_df.set_index("item_id").loc[ranked_ids].copy()
        result["score_hybrid"] = [score for _, score in ranked]
        return result.reset_index()

    def generate_ministore_html_from_item_ids(
        self,
        item_ids: List[str],
        title: str = "Productos relacionados",
    ) -> str:
        if not item_ids:
            return ""

        df = self.ads_df.set_index("item_id").loc[item_ids].copy()

        cards_html = []
        for _, row in df.iterrows():
            t = row.get("title", "")
            d = row.get("description", "")
            url = row.get("url", "") or "#"
            kw = row.get("keywords", "")
            lang = row.get("language", "")

            card = f"""
            <div class="mini-card" style="
                flex: 0 0 220px;
                border-radius: 16px;
                padding: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                background: #ffffff;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <h3 style="font-size: 1rem; margin: 0 0 4px 0;">{t}</h3>
                    <p style="font-size: 0.85rem; margin: 0 0 6px 0; color:#333;">{d}</p>
                    <p style="font-size: 0.75rem; margin: 0; color:#777;">
                        <strong>Keywords:</strong> {kw}
                    </p>
                    <p style="font-size: 0.7rem; margin: 4px 0 0 0; color:#999;">
                        {lang}
                    </p>
                </div>
                <div style="margin-top: 10px; display:flex; justify-content:flex-end; align-items:center;">
                    <a href="{url}" target="_blank" rel="noopener noreferrer"
                       style="font-size:0.85rem; padding:6px 10px; border-radius:999px;
                              background:#2563eb; color:white; text-decoration:none;">
                        Ver producto
                    </a>
                </div>
            </div>
            """
            cards_html.append(card)

        ministore_html = f"""
        <section class="ministore" style="margin-top: 24px;">
            <h2 style="font-size: 1.25rem; margin-bottom: 12px;">{title}</h2>
            <div class="ministore-grid" style="
                display:flex;
                gap: 12px;
                overflow-x:auto;
                padding-bottom: 8px;
            ">
                {''.join(cards_html)}
            </div>
        </section>
        """
        return ministore_html.strip()


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


def refresh_ministore_items_from_serper(
    db: MySQLConnector,
    query: str,
    num_results: int = 10,
    language: str = "es",
    table_name: str = "ministore_items",
) -> int:
    items_df = fetch_ministore_items_from_serper(
        query=query,
        num_results=num_results,
        language=language,
    )
    return upsert_ministore_items_into_db(
        db=db,
        items_df=items_df,
        table_name=table_name,
    )


def generate_ministore_for_article_from_db(
    db: MySQLConnector,
    article_text: str,
    items_table: str = "ministore_items",
    interactions_table: Optional[str] = "ministore_interactions",
    user_id: Optional[str] = None,
    anchor_item_id: Optional[str] = None,
    top_k: int = 5,
    summarize: bool = True,
    summary_max_chars: int = 2000,
    title: str = "Productos relacionados con esta noticia",
) -> str:
    items_df = load_ministore_items_from_db(db=db, table_name=items_table)
    rec = OpenAIAdRecommender(items_df)

    if interactions_table is not None:
        interactions_df = load_user_interactions_from_db(db=db, table_name=interactions_table)
        if not interactions_df.empty:
            rec.fit_user_interactions(interactions_df)

    if user_id is not None and anchor_item_id is not None:
        hybrid_df = rec.recommend_items_hybrid(
            user_id=str(user_id),
            anchor_item_id=str(anchor_item_id),
            top_n=top_k,
        )
        item_ids = hybrid_df["item_id"].tolist()
        return rec.generate_ministore_html_from_item_ids(item_ids, title=title)

    recommendations = rec.analyze_article(
        raw_article_text=article_text,
        top_k=top_k,
        summarize=summarize,
        summary_max_chars=summary_max_chars,
    )
    item_ids = recommendations["item_id"].tolist()
    return rec.generate_ministore_html_from_item_ids(item_ids, title=title)


def summarize_article_and_ministores(
    db: MySQLConnector,
    article_text: str,
    items_table: str = "ministore_items",
    interactions_table: Optional[str] = "ministore_interactions",
    user_id: Optional[str] = None,
    num_ministores: int = 3,
    items_per_ministore: int = 4,
    base_ministore_url: Optional[str] = None,
    summary_max_chars: int = 2000,
    title_prefix: str = "Productos relacionados",
) -> Dict[str, Any]:
    if not article_text or not article_text.strip():
        raise ValueError("Article text is empty.")
    article_text = article_text.strip()

    summary_text = summarize_spanish_article(
        article_text=article_text,
        max_chars=summary_max_chars,
    )

    items_df = load_ministore_items_from_db(db=db, table_name=items_table)
    rec = OpenAIAdRecommender(items_df)

    if interactions_table is not None:
        interactions_df = load_user_interactions_from_db(db=db, table_name=interactions_table)
        if not interactions_df.empty and user_id is not None:
            rec.fit_user_interactions(interactions_df)

    total_items_needed = num_ministores * items_per_ministore
    recs_df = rec.analyze_keywords(
        article_text=summary_text,
        top_k=total_items_needed,
    )

    item_ids_all = recs_df["item_id"].tolist()
    ministores: List[Dict[str, Any]] = []

    def _escape_for_srcdoc(html: str) -> str:
        return html.replace("&", "&amp;").replace("\"", "&quot;")

    for i in range(num_ministores):
        start = i * items_per_ministore
        end = start + items_per_ministore
        chunk_ids = item_ids_all[start:end]
        if not chunk_ids:
            break
        title = f"{title_prefix} #{i + 1}"
        html = rec.generate_ministore_html_from_item_ids(chunk_ids, title=title)

        if base_ministore_url is not None:
            url = f"{base_ministore_url}?slot={i + 1}"
            iframe = f'<iframe src="{url}" style="width:100%;border:none;overflow:hidden;"></iframe>'
        else:
            url = None
            iframe = f'<iframe srcdoc="{_escape_for_srcdoc(html)}" style="width:100%;border:none;overflow:hidden;"></iframe>'

        ministores.append(
            {
                "index": i + 1,
                "item_ids": chunk_ids,
                "url": url,
                "html": html,
                "iframe": iframe,
            }
        )

    return {
        "summary": summary_text,
        "ministores": ministores,
    }


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
