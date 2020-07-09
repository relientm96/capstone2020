import requests
from bs4 import BeautifulSoup

import pprint as pp

def getWordTitle(soup):
    """
    Gets the Current Word for this URL from SignBank Website
    """
    return soup.find(id="keywords").find("p").find("b").get_text()

def getLink(soup):
    """
    Get's Video link from a given beautiful soup URL from SignBank Website
    """
    return soup.find("video").find("source").get("src")

def main():
    urls = [
    "http://www.auslan.org.au/dictionary/words/one-1.html",
    "http://www.auslan.org.au/dictionary/words/two-1.html",
    "http://www.auslan.org.au/dictionary/words/three-1.html",
    "http://www.auslan.org.au/dictionary/words/four-1.html",
    "http://www.auslan.org.au/dictionary/words/five-1.html",
    "http://www.auslan.org.au/dictionary/words/six-1.html",
    "http://www.auslan.org.au/dictionary/words/seven-1.html",
    "http://www.auslan.org.au/dictionary/words/eight-1.html",
    "http://www.auslan.org.au/dictionary/words/nine-1.html",
    "http://www.auslan.org.au/dictionary/words/ten-1.html",
    ]

    """
    for link in urls:
        # Use Beautiful Soup to get HTML page and extract mp4 links
        req = requests.get(link)
        soup = BeautifulSoup(req.content, 'html.parser')
        pp.pprint(getLink(soup))
    """
    req = requests.get("http://www.auslan.org.au/dictionary/words/one-1.html")
    soup = BeautifulSoup(req.content, 'html.parser')
    print(getWordTitle(soup),",",getLink(soup))

if __name__ == "__main__":
    print("Starting Sign Bank Scrapper Program")
    main()
