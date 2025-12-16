# summarizer.py
import os
import re
import urllib.parse
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_deanna_ministore(search_query: str) -> str:
    encoded_query = urllib.parse.quote(search_query)
    return f"https://www.deanna2u.com/?q={encoded_query}"


def summarize_article_overall(
    article_text: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Create a short overall summary of the article (Spanish), 2–3 sentences.
    """
    if not article_text or not article_text.strip():
        raise ValueError("El texto del artículo está vacío.")

    trimmed_text = article_text.strip()
    if len(trimmed_text) > 15000:
        trimmed_text = trimmed_text[:15000]

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un periodista. Resume el artículo de forma clara y neutral.\n"
                "Devuelve SOLO el resumen en español, en 2-3 frases, sin títulos, sin viñetas."
            ),
        },
        {
            "role": "user",
            "content": f"ARTÍCULO:\n{trimmed_text}",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    summary = response.choices[0].message.content.strip()
    # Keep it tight
    if len(summary) > 500:
        summary = summary[:500].rstrip() + "…"
    return summary


def summarize_spanish_article(
    article_text: str,
    model: str = "gpt-4o-mini",
    max_chars: Optional[int] = 5000,
) -> list[str]:
    """
    Extract two commercial/ad-friendly topics from a Spanish article.
    (UNCHANGED)
    """
    if not article_text or not article_text.strip():
        raise ValueError("El texto del artículo está vacío.")

    trimmed_text = article_text.strip()
    if len(trimmed_text) > 15000:
        trimmed_text = trimmed_text[:15000]

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un experto en marketing digital especializado en identificar oportunidades "
                "comerciales y publicitarias en artículos periodísticos.\n\n"
                "Tu objetivo es extraer dos temas comerciales del artículo que sean perfectos "
                "para generar anuncios o contenido publicitario relacionado.\n\n"
                "FORMATO DE RESPUESTA:\n"
                "Debes devolver EXACTAMENTE dos búsquedas comerciales, una por línea, sin numeración ni viñetas.\n"
                "Cada búsqueda debe tener MÁXIMO 5 palabras.\n"
                "Cada búsqueda debe ser específica, comercial y útil para generar anuncios relevantes.\n"
                "Las búsquedas deben ser diferentes entre sí y enfocarse en productos, servicios, "
                "lugares o actividades mencionadas en el artículo.\n\n"
                "EJEMPLOS de buenos temas comerciales:\n"
                "mejores restaurantes Madrid centro\n"
                "hoteles económicos Barcelona playa\n"
                "cursos online marketing digital\n"
                "smartphones gama media 2024\n"
                "gimnasios cerca de mí"
            ),
        },
        {
            "role": "user",
            "content": (
                "Analiza el siguiente artículo periodístico e identifica DOS temas comerciales "
                "que sean perfectos para generar anuncios o buscar productos/servicios relacionados.\n\n"
                "INSTRUCCIONES:\n"
                "- Lee el artículo completo y comprende su contenido.\n"
                "- Identifica dos aspectos del artículo que tengan potencial comercial o publicitario.\n"
                "- Piensa en qué productos, servicios, lugares o actividades podrían interesar a alguien "
                "que lee este artículo.\n"
                "- Cada tema debe ser una búsqueda comercial de máximo 5 palabras que sea específica "
                "y útil para encontrar anuncios relevantes.\n"
                "- Usa términos que alguien escribiría en un buscador para encontrar productos o servicios.\n"
                "- Las búsquedas deben ser distintas entre sí.\n"
                "- Puedes usar español o inglés dependiendo de lo que sea más natural para el tema.\n"
                "- Devuelve EXACTAMENTE dos líneas, cada una con una búsqueda comercial "
                "(sin numeración, sin viñetas, sin explicaciones adicionales).\n\n"
                "ARTÍCULO A ANALIZAR:\n"
                f"{trimmed_text}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )

    topics_text = response.choices[0].message.content.strip()
    topics = [line.strip() for line in topics_text.split('\n') if line.strip()]
    topics = [t for t in topics if not t.startswith(('Ejemplo', 'Formato', 'INSTRUCCIONES', 'ARTÍCULO', 'EJEMPLOS'))]
    topics = [re.sub(r'^\d+[\.\)]\s*', '', t) for t in topics]

    if len(topics) < 2:
        if len(topics) == 1:
            parts = topics[0].split('|')
            if len(parts) >= 2:
                topics = [p.strip() for p in parts[:2]]
            else:
                topics = [topics[0], topics[0]]
        else:
            topics = ["productos relacionados", "servicios disponibles"]
    elif len(topics) > 2:
        topics = topics[:2]

    validated_topics = []
    for topic in topics:
        words = topic.split()
        if len(words) > 5:
            topic = ' '.join(words[:5])
        validated_topics.append(topic)

    return validated_topics
