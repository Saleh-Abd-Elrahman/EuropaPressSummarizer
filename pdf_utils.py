# pdf_utils.py
from typing import BinaryIO

from PyPDF2 import PdfReader


def extract_text_from_pdf(file: BinaryIO) -> str:
    """
    Extracts text from a PDF file-like object.

    Parameters
    ----------
    file : BinaryIO
        A file-like object for the uploaded PDF.

    Returns
    -------
    str
        Extracted text.
    """
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        texts.append(page_text)

    return "\n".join(texts)
