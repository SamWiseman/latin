import requests
import json
import csv
import re
#Adam Klein, 10/18/2017
#run with Python2

def main():
    #the title of the page (as it appears in the url)
    page = "Bede"
    #the number of links to scrape
    linksNumber = "20"
    
    #the url of the page to trawl for links.
    url = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=info&inprop=url&titles=" + page + "&generator=links&gpllimit=" + linksNumber
    response = requests.get(url)
    with open('wiki.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile)
        for pageInfo in response.json()["query"]["pages"].itervalues():
            pageTitle = pageInfo["title"].encode('ascii', 'ignore')
            pageId = str(pageInfo["pageid"])
            page = requests.get("https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=text&pageid=" + pageId)
            pageHtml = page.json()["parse"]["text"]["*"]
            #parse out html tags and nonletters
            for word in re.compile(r'<.*?>|[^a-zA-Z ]').sub('', pageHtml).encode('ascii', 'ignore').split():
                csvwriter.writerow([word.lower(), pageTitle])
    

if __name__ == "__main__":
    main()