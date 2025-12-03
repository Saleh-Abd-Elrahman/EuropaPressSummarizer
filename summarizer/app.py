# app.py
import streamlit as st

from pdf_utils import extract_text_from_pdf
from web_utils import fetch_article_text_from_url
from summarizer import summarize_spanish_article
from storage import save_summary, load_all_summaries


st.set_page_config(
    page_title="Resumen de Artículos en Español",
    page_icon="📝",
    layout="wide",
)


st.title("📝 Resumen de Artículos en Español")
st.write(
    "Sube un PDF en español o introduce un enlace a un artículo, "
    "y este sistema lo resumirá usando el modelo de OpenAI (GPT)."
)

tab_pdf, tab_url, tab_history = st.tabs(["📄 PDF", "🌐 URL", "📚 Historial"])


# --------- PDF TAB ---------
with tab_pdf:
    st.subheader("Subir PDF")

    uploaded_pdf = st.file_uploader(
        "Selecciona un archivo PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_pdf is not None:
        st.info(f"Archivo seleccionado: {uploaded_pdf.name}")

        if st.button("Resumir PDF"):
            with st.spinner("Extrayendo texto del PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)

            if not pdf_text.strip():
                st.error("No se ha podido extraer texto del PDF.")
            else:
                st.success("Texto extraído correctamente. Generando resumen...")
                with st.spinner("Llamando a OpenAI para generar el resumen en español..."):
                    try:
                        summary = summarize_spanish_article(pdf_text)
                        st.subheader("Resumen generado (PDF)")
                        st.write(summary)

                        record = save_summary(
                            source_type="pdf",
                            source_name=uploaded_pdf.name,
                            summary=summary,
                            language="es",
                        )

                        st.success("Resumen guardado correctamente.")
                        st.caption(f"ID del resumen: {record.id}")

                    except Exception as e:
                        st.error(f"Error generando el resumen: {e}")


# --------- URL TAB ---------
with tab_url:
    st.subheader("Introducir URL de artículo")

    url = st.text_input("Pega aquí el enlace del artículo")

    if st.button("Resumir URL"):
        if not url.strip():
            st.error("Por favor, introduce una URL válida.")
        else:
            with st.spinner("Descargando y extrayendo el contenido del artículo..."):
                try:
                    article_text = fetch_article_text_from_url(url)
                except Exception as e:
                    st.error(f"No se ha podido obtener el artículo: {e}")
                    article_text = ""

            if not article_text.strip():
                st.error("No se ha podido extraer texto del artículo.")
            else:
                st.success("Texto del artículo obtenido. Generando resumen...")
                with st.spinner("Llamando a OpenAI para generar el resumen en español..."):
                    try:
                        summary = summarize_spanish_article(article_text)
                        st.subheader("Resumen generado (URL)")
                        st.write(summary)

                        record = save_summary(
                            source_type="url",
                            source_name=url,
                            summary=summary,
                            language="es",
                        )

                        st.success("Resumen guardado correctamente.")
                        st.caption(f"ID del resumen: {record.id}")

                    except Exception as e:
                        st.error(f"Error generando el resumen: {e}")


# --------- HISTORY TAB ---------
with tab_history:
    st.subheader("Historial de resúmenes guardados")

    summaries = load_all_summaries()
    if not summaries:
        st.info("Todavía no hay resúmenes guardados.")
    else:
        for rec in reversed(summaries):  # most recent first
            with st.expander(f"{rec.source_type.upper()} - {rec.source_name} ({rec.created_at})"):
                st.write(rec.summary)
                st.caption(f"ID: {rec.id} | Idioma: {rec.language}")
