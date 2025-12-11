# summarizer.py

#TODO: Update summarizer to instead 1) Extract the two main interesting parts of the article and 2) Use those topics to generate mini stores to display on the page (2 per article).


import os
import re
import urllib.parse
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_deanna_ministore(search_query: str) -> str:
    """
    Create a mini store on deanna2u.com using a search query.
    
    Parameters
    ----------
    search_query : str
        The search term to create a mini store for.
        
    Returns
    -------
    str
        The URL of the created mini store.
    """
    # Encode the search query for URL
    encoded_query = urllib.parse.quote(search_query)
    
    # Create the deanna2u.com search URL
    # Based on the website structure, searches can be performed via URL parameters
    ministore_url = f"https://www.deanna2u.com/?q={encoded_query}"
    
    return ministore_url


def summarize_spanish_article(
    article_text: str,
    model: str = "gpt-4o-mini",
    max_chars: Optional[int] = 5000,
) -> list[str]:
    """
    Extract two commercial/ad-friendly topics from a Spanish article.

    Parameters
    ----------
    article_text : str
        Full article text (in Spanish or any language).
    model : str
        OpenAI model name.
    max_chars : int, optional
        Not used, kept for backward compatibility.

    Returns
    -------
    list[str]
        A list of exactly two commercial topics (max 5 words each).
    """

    if not article_text or not article_text.strip():
        raise ValueError("El texto del artículo está vacío.")

    # Optional: truncate very long texts (avoid token explosion)
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
        temperature=0.4,  # slightly higher for more creative commercial ideas
    )

    topics_text = response.choices[0].message.content.strip()
    
    # Parse the response to extract exactly two topics
    topics = [line.strip() for line in topics_text.split('\n') if line.strip()]
    
    # Filter out any lines that look like explanations or formatting
    topics = [t for t in topics if not t.startswith(('Ejemplo', 'Formato', 'INSTRUCCIONES', 'ARTÍCULO', 'EJEMPLOS'))]
    
    # Remove numbering if present (e.g., "1. " or "1) ")
    topics = [re.sub(r'^\d+[\.\)]\s*', '', t) for t in topics]
    
    # Ensure we have exactly 2 topics
    if len(topics) < 2:
        # If we got fewer than 2, try splitting by other delimiters or take what we have
        if len(topics) == 1:
            # Try splitting by common separators
            parts = topics[0].split('|')
            if len(parts) >= 2:
                topics = [p.strip() for p in parts[:2]]
            else:
                # If still only one, duplicate it (fallback)
                topics = [topics[0], topics[0]]
        else:
            # Fallback if no topics found
            topics = ["productos relacionados", "servicios disponibles"]
    elif len(topics) > 2:
        # Take only the first two
        topics = topics[:2]
    
    # Validate each topic is max 5 words
    validated_topics = []
    for topic in topics:
        words = topic.split()
        if len(words) > 5:
            # Truncate to 5 words
            topic = ' '.join(words[:5])
        validated_topics.append(topic)
    
    return validated_topics
