# ministore_books.py
import datetime
from typing import List

from MySQLConnector import MySQLConnector
from ministore_engine import fetch_ministore_items_from_serper


def _book_url(base_book_url: str, slug: str) -> str:
    base = (base_book_url or "").rstrip("/")
    return f"{base}/{slug}"


def create_book_from_topic(
    db: MySQLConnector,
    topic: str,
    user_id: int,
    category_id: int,
    language: str,
    base_book_url: str,
    items_per_book: int = 4,
    serper_num_results: int = 10,
    slug_prefix: str = "ministore",
) -> str:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic is empty")

    # 1) fetch items from Serper
    items_df = fetch_ministore_items_from_serper(
        query=topic,
        num_results=serper_num_results,
        language=language,
    )

    # take only items_per_book
    items_df = items_df.head(items_per_book)

    # 2) create cliperest_book row
    now = datetime.datetime.now()
    fecha_str_day = now.strftime("%d/%m/%Y")
    created_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    slug_ts = now.strftime("%d-%m-%Y-%H%M%S")
    fecha_str_americana = now.strftime("%Y-%m-%d %H:%M:%S")

    slug = f"{slug_prefix}-{slug_ts}"
    name = f"{topic} - {fecha_str_day}"
    description = f"Ministore auto-generado para: {topic}"

    book_data = {
        "user_id": 221,
        "name": name,
        "slug": slug,
        "rendered": 0,
        "version": 1,
        "category_id": 30,
        "modified": fecha_str_americana,
        "addEnd": 1,
        "coverImage": "/img/covers/default_newcover.jpg",
        "sharing": 0,
        "coverColor": 2,
        "dollarsGiven": 0,
        "privacy": 0,
        "type": 0,
        "created": fecha_str_americana,
        "coverHexColor": "#336699",
        "numLikers": 0,
        "description": description,
        "tags": "",
        "thumbnailImage": "",
        "numClips": 0,
        "numViews": 0,
        "userLanguage": "es-ES",
        "embed_code": None,
        "thumbnailImageSmall": "",
        "humanModified": fecha_str_americana,
        "coverV3": 1,
        "typeFilters": "a:0:{}",
    }

    book_id = db.create_book(book_data)
    if not book_id:
        raise RuntimeError("Failed to create book in DB")

    # 3) create clippings (cliperest_clipping)
    clippings = []
    for i, row in enumerate(items_df.to_dict(orient="records"), start=1):
        caption = (row.get("title") or "Producto relacionado").strip()
        text = (row.get("description") or "").strip()
        url = (row.get("url") or "").strip()

        clippings.append(
            {
                "book_id": book_id,
                "caption": caption,
                "text": text,
                "thumbnail": "",
                "useThumbnail": 0,
                "type": 0,
                "url": url,
                "created": created_ts,
                "num": i,
                "migratedS3": 0,
                "modified": created_ts,
            }
        )

    inserted = db.create_clippings_batch(clippings)
    # optional: update numClips if your schema uses it (safe even if you ignore)
    try:
        db.execute_query("UPDATE cliperest_book SET numClips = %s WHERE id = %s", (inserted or 0, book_id))
    except Exception:
        pass

    return _book_url(base_book_url, slug)


def create_three_books_for_topics(
    topics: List[str],
    user_id: int = 221,
    category_id: int = 1,
    language: str = "es",
    base_book_url: str = "https://www.deanna2u.com/book",
    items_per_book: int = 4,
    serper_num_results: int = 10,
) -> List[str]:
    if not topics or len(topics) != 3:
        raise ValueError("topics must be a list of exactly 3 strings")

    db = MySQLConnector()
    db.connect()
    if not db.connection or not db.connection.is_connected():
        raise RuntimeError("MySQL connection failed. Check DB_* env vars.")

    try:
        urls = []
        for idx, topic in enumerate(topics, start=1):
            url = create_book_from_topic(
                db=db,
                topic=topic,
                user_id=user_id,
                category_id=category_id,
                language=language,
                base_book_url=base_book_url,
                items_per_book=items_per_book,
                serper_num_results=serper_num_results,
                slug_prefix=f"ministore-{idx}",
            )
            urls.append(url)
        return urls
    finally:
        try:
            db.disconnect()
        except Exception:
            pass
