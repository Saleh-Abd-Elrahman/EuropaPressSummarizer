# summarizer.py
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_spanish_article(
    article_text: str,
    model: str = "gpt-4o-mini",
    max_chars: Optional[int] = 5000,
) -> str:
    """
    Summarize a Spanish article using OpenAI's GPT model.

    Parameters
    ----------
    article_text : str
        Full article text (in Spanish or any language).
    model : str
        OpenAI model name.
    max_chars : int, optional
        Optional character limit for the summary (approximate).

    Returns
    -------
    str
        A grammatically correct Spanish summary.
    """

    if not article_text or not article_text.strip():
        raise ValueError("El texto del artículo está vacío.")

    # Optional: truncate very long texts (avoid token explosion)
    trimmed_text = article_text.strip()
    if len(trimmed_text) > 15000:
        trimmed_text = trimmed_text[:15000]

    # System + user messages to make sure summary is in proper Spanish
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente experto en resumir artículos periodísticos en "
                "español claro, conciso y con gramática correcta. "
                "Devuelves sólo el resumen en español, sin comentarios adicionales."
            ),
        },
        {
            "role": "user",
            "content": (
                "Resume el siguiente artículo en español. "
                "Debes producir un texto cohesivo, bien estructurado, y con buena "
                "sintaxis. Evita listas; usa párrafos completos.\n\n"
                f"Artículo:\n{trimmed_text}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # lower temperature = more deterministic / coherent
    )

    summary = response.choices[0].message.content.strip()

    if max_chars and len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."

    return summary
