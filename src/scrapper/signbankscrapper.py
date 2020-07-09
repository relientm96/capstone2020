import requests
from bs4 import BeautifulSoup

url = "http://www.auslan.org.au/dictionary/words/one-1.html"
req = requests.get(url)
soup = BeautifulSoup(req.content, 'html.parser')
print(soup.prettify())