import requests
import urllib.request
from bs4 import BeautifulSoup
import pprint as pp
import os
from pathlib import Path

# Import file holding all urls in lists
from urls import *

# Current Folder Path
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

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

    # Before scrapping, make directories first to save videos
    try:
        os.makedirs("videos")
    except FileExistsError:
        # We skip if directories already exist
        pass
    # Get video directory by appending videos to current directory path
    videoDirectory = os.path.join(THIS_FOLDER,"videos")
    
    for link in urlList:
        # Use Requests and Beautiful Soup to get HTML page
        req = requests.get(link)
        soup = BeautifulSoup(req.content, 'html.parser')

        ### Extract MP4 Links and Sign Word from links ###
        # Get word from html and remove white spaces
        signWord = getWordTitle(soup).strip()
        # Extract video link
        vidLink  = getLink(soup)

        print("{},{}".format(signWord,vidLink))
        
        ### Download Video from extracted URL into local directory ###
        # Extract video format
        vidFormat = "." + vidLink.split('.')[-1] 
        # Create video file name by joining word + format
        vidFilename = signWord + vidFormat
        # File path to save downloading video
        vidFilePath = os.path.join(videoDirectory, vidFilename)
        # Download and save video
        urllib.request.urlretrieve(vidLink, vidFilePath) 

if __name__ == "__main__":
    print("Starting Sign Bank Scrapper Program")
    main()
