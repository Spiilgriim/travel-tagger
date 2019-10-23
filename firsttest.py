import bs4
import urllib.request

url = "https://www.votretourdumonde.com/bruxelles-insolite-10-choses-decouvrir/"
req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"}) 
con = urllib.request.urlopen( req )
webpage = str(con.read().decode("utf-8"))
soup = bs4.BeautifulSoup(webpage, "lxml")

print(soup.title.get_text())
for poccur in soup.find_all('p'):
    print(poccur.get_text())