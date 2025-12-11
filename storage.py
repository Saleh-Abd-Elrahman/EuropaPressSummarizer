# storage.py
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SUMMARIES_FILE = DATA_DIR / "summaries.jsonl"


@dataclass
class SummaryRecord:
    id: str
    source_type: str  # "pdf" or "url"
    source_name: str  # filename or URL
    language: str
    created_at: str
    summary: str


def save_summary(
    source_type: str,
    source_name: str,
    summary: str,
    language: str = "es",
) -> SummaryRecord:
    """
    Save a summary to a JSONL file and return the record.

    Parameters
    ----------
    source_type : str
        "pdf" or "url"
    source_name : str
        Filename or URL
    summary : str
        The summary text
    language : str
        Language code, default "es" for Spanish.
    """
    now = datetime.utcnow().isoformat()
    record_id = f"{source_type}-{now}"

    record = SummaryRecord(
        id=record_id,
        source_type=source_type,
        source_name=source_name,
        language=language,
        created_at=now,
        summary=summary,
    )

    with SUMMARIES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    return record


def load_all_summaries() -> List[SummaryRecord]:
    """
    Load all summary records from the JSONL file.
    """
    records: List[SummaryRecord] = []
    if not SUMMARIES_FILE.exists():
        return records

    with SUMMARIES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(SummaryRecord(**data))
    return records
