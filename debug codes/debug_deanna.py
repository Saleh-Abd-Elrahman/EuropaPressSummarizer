import requests

url = "https://deanna.today/andres-gonzalez-alvarado-tomara-posesion-el-9-de-diciembre-como-jefe-de-la-brilat/"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

resp = requests.get(url, headers=headers, timeout=15)
print("STATUS:", resp.status_code)
print("FIRST 500 CHARS:\n", resp.text[:500])
