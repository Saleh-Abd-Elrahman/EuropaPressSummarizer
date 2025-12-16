# ministore_creator.py
import time
import uuid
from dataclasses import dataclass
from typing import List

from MySQLConnector import MySQLConnector
from ministore_engine import (
    fetch_ministore_items_from_serper,
    upsert_ministore_items_into_db,
)

@dataclass
class MinistoreCreateResult:
    ministore_id: str
    topic: str
    item_ids: List[str]


def get_db() -> MySQLConnector:
    db = MySQLConnector()
    db.connect()
    if not db.connection or not db.connection.is_connected():
        raise RuntimeError("MySQL connection failed. Check DB_* env vars.")
    return db


def ensure_tables(db: MySQLConnector) -> None:
    db.execute_query(
        """
        CREATE TABLE IF NOT EXISTS ministores (
            id VARCHAR(64) PRIMARY KEY,
            topic VARCHAR(255) NOT NULL,
            language VARCHAR(8) NOT NULL DEFAULT 'es',
            created_at BIGINT NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    db.execute_query(
        """
        CREATE TABLE IF NOT EXISTS ministore_items (
            id VARCHAR(128) PRIMARY KEY,
            title TEXT,
            description TEXT,
            url TEXT,
            keywords VARCHAR(255),
            language VARCHAR(8)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    db.execute_query(
        """
        CREATE TABLE IF NOT EXISTS ministore_item_map (
            ministore_id VARCHAR(64) NOT NULL,
            item_id VARCHAR(128) NOT NULL,
            pos INT NOT NULL,
            PRIMARY KEY (ministore_id, item_id),
            KEY idx_ministore (ministore_id),
            CONSTRAINT fk_ministore
              FOREIGN KEY (ministore_id) REFERENCES ministores(id)
              ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )


def create_ministore_in_db(
    db: MySQLConnector,
    topic: str,
    language: str = "es",
    num_results: int = 10,
    items_to_link: int = 8,
) -> MinistoreCreateResult:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("topic is empty")

    ensure_tables(db)

    items_df = fetch_ministore_items_from_serper(
        query=topic,
        num_results=num_results,
        language=language,
    )

    upsert_ministore_items_into_db(db=db, items_df=items_df, table_name="ministore_items")

    ministore_id = uuid.uuid4().hex
    created_at = int(time.time())

    db.execute_query(
        "INSERT INTO ministores (id, topic, language, created_at) VALUES (%s, %s, %s, %s)",
        (ministore_id, topic, language, created_at),
    )

    item_ids = [str(x) for x in items_df["id"].tolist()][:items_to_link]

    if item_ids:
        values = [(ministore_id, item_id, idx) for idx, item_id in enumerate(item_ids)]
        sql = "INSERT IGNORE INTO ministore_item_map (ministore_id, item_id, pos) VALUES (%s, %s, %s)"

        cursor = None
        try:
            cursor = db.connection.cursor()
            cursor.executemany(sql, values)
            db.connection.commit()
        finally:
            if cursor:
                cursor.close()

    return MinistoreCreateResult(ministore_id=ministore_id, topic=topic, item_ids=item_ids)


def render_ministore_html_from_db(db: MySQLConnector, ministore_id: str) -> str:
    """
    Build a simple public HTML page from the DB rows.
    """
    ensure_tables(db)

    ms = db.execute_query("SELECT id, topic, language, created_at FROM ministores WHERE id=%s LIMIT 1", (ministore_id,))
    if not ms:
        return ""

    topic = ms[0]["topic"]

    rows = db.execute_query(
        """
        SELECT i.title, i.description, i.url, i.keywords, i.language
        FROM ministore_item_map m
        JOIN ministore_items i ON i.id = m.item_id
        WHERE m.ministore_id = %s
        ORDER BY m.pos ASC
        LIMIT 50
        """,
        (ministore_id,),
    ) or []

    cards = []
    for r in rows:
        title = (r.get("title") or "").strip()
        desc = (r.get("description") or "").strip()
        url = (r.get("url") or "#").strip()
        cards.append(f"""
          <div style="flex:0 0 260px;background:#fff;border-radius:16px;padding:12px;box-shadow:0 2px 10px rgba(0,0,0,.08);">
            <div style="font-weight:700;margin-bottom:6px;">{title}</div>
            <div style="font-size:13px;color:#374151;line-height:1.35;margin-bottom:10px;">{desc}</div>
            <a href="{url}" target="_blank" rel="noopener" style="display:inline-block;background:#2563eb;color:#fff;padding:8px 10px;border-radius:999px;text-decoration:none;font-size:13px;">
              Ver producto
            </a>
          </div>
        """)

    cards_html = "\n".join(cards) if cards else "<div style='color:#6b7280;'>No items.</div>"

    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Ministore - {topic}</title>
</head>
<body style="margin:0;font-family:ui-sans-serif,system-ui;background:#0b1220;color:#e5e7eb;">
  <div style="max-width:1100px;margin:0 auto;padding:24px;">
    <div style="font-size:22px;font-weight:800;margin-bottom:6px;color:#f97316;">Productos relacionados</div>
    <div style="font-size:14px;color:#a5b4fc;margin-bottom:16px;">Tema: {topic}</div>

    <div style="display:flex;gap:14px;overflow-x:auto;padding-bottom:10px;">
      {cards_html}
    </div>
  </div>
</body>
</html>
""".strip()
