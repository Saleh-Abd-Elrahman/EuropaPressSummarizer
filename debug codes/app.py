# app.py
import streamlit as st

from pdf_utils import extract_text_from_pdf
from web_utils import fetch_article_text_from_url
from summarizer import summarize_spanish_article
from storage import save_summary, load_all_summaries


st.set_page_config(
    page_title="Identificador de Temas Comerciales",
    page_icon="🎯",
    layout="wide",
)


st.title("🎯 Identificador de Temas Comerciales")
st.write(
    "Sube un PDF en español o introduce un enlace a un artículo, "
    "y este sistema identificará dos temas comerciales relevantes para publicidad y marketing."
)

tab_pdf, tab_url, tab_history = st.tabs(["📄 PDF", "🌐 URL", "📚 Historial de Temas"])


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

        if st.button("Identificar Temas Comerciales del PDF"):
            with st.spinner("Extrayendo texto del PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)

            if not pdf_text.strip():
                st.error("No se ha podido extraer texto del PDF.")
            else:
                st.success("Texto extraído correctamente. Generando temas comerciales...")
                with st.spinner("Llamando a OpenAI para identificar temas comerciales..."):
                    try:
                        topics = summarize_spanish_article(pdf_text)
                        st.subheader("🎯 Temas Comerciales Identificados (PDF)")
                        st.write("Se han identificado dos temas comerciales del artículo:")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Tema 1:**")
                            st.info(topics[0])
                        with col2:
                            st.markdown(f"**Tema 2:**")
                            st.info(topics[1])

                        # Save the topics as summary
                        summary_text = f"Tema 1: {topics[0]}\nTema 2: {topics[1]}"
                        record = save_summary(
                            source_type="pdf",
                            source_name=uploaded_pdf.name,
                            summary=summary_text,
                            language="es",
                        )

                        st.success("Temas comerciales identificados y guardados correctamente.")
                        st.caption(f"ID del registro: {record.id}")

                    except Exception as e:
                        st.error(f"Error identificando los temas comerciales: {e}")


# --------- URL TAB ---------
with tab_url:
    st.subheader("Introducir URL de artículo")

    url = st.text_input("Pega aquí el enlace del artículo")

    if st.button("Identificar Temas Comerciales del URL"):
        if not url.strip():
            st.error("Por favor, introduce una URL válida.")
        else:
            article_text = None
            try:
                with st.spinner("Descargando y extrayendo el contenido del artículo..."):
                    article_text = fetch_article_text_from_url(url)
            except Exception as e:
                st.error(f"No se ha podido obtener el artículo: {e}")

            if article_text:
                st.success("Texto del artículo obtenido. Generando temas comerciales...")
                with st.spinner("Llamando a OpenAI para identificar temas comerciales..."):
                    try:
                        topics = summarize_spanish_article(article_text)
                        st.subheader("🎯 Temas Comerciales Identificados (URL)")
                        st.write("Se han identificado dos temas comerciales del artículo:")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Tema 1:**")
                            st.info(topics[0])
                        with col2:
                            st.markdown(f"**Tema 2:**")
                            st.info(topics[1])

                        # Save the topics as summary
                        summary_text = f"Tema 1: {topics[0]}\nTema 2: {topics[1]}"
                        record = save_summary(
                            source_type="url",
                            source_name=url,
                            summary=summary_text,
                            language="es",
                        )

                        st.success("Temas comerciales identificados y guardados correctamente.")
                        st.caption(f"ID del registro: {record.id}")

                    except Exception as e:
                        st.error(f"Error identificando los temas comerciales: {e}")



# --------- HISTORY TAB ---------
with tab_history:
    st.subheader("Historial de temas comerciales identificados")

    summaries = load_all_summaries()
    if not summaries:
        st.info("Todavía no hay temas identificados guardados.")
    else:
        for rec in reversed(summaries):  # most recent first
            with st.expander(f"{rec.source_type.upper()} - {rec.source_name} ({rec.created_at})"):
                st.write(rec.summary)
                st.caption(f"ID: {rec.id} | Idioma: {rec.language}")
