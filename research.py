
import requests, bs4
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def ddg_search(query, n=5):
    html = requests.get('https://duckduckgo.com/html/?q='+query, headers=HEADERS, timeout=10).text
    soup = bs4.BeautifulSoup(html,'html.parser')
    return [a.get_text(' ', strip=True) for a in soup.select('.result__a')[:n]]
