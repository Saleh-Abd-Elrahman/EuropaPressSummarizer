from web_utils import fetch_article_text_from_url

url = "https://deanna.today/andres-gonzalez-alvarado-tomara-posesion-el-9-de-diciembre-como-jefe-de-la-brilat/"

text = fetch_article_text_from_url(url)
print("LEN:", len(text))
print("FIRST 800 CHARS:\n", text[:8000])
