import bs4
import urllib.request
import time as t
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas

url = "https://www.votretourdumonde.com"
banlist = ['#comment']
nlp = spacy.load('fr_core_news_sm')
text_list = []

def explore_website(url, banlist, text_list):
    i=0
    url_list = [url]
    visited_list = []
    initialt = t.time()
    while url_list != []:
        actual_url = url_list.pop(0)
        visited_list.append(actual_url)
        print(i, actual_url, t.time()-initialt, "s")
        initialt = t.time()
        i += 1
        cpass = False
        for x in banlist:
            if actual_url.find(x) != -1:
                cpass = True
        if not cpass:
            try: 
                req = urllib.request.Request(actual_url, headers={'User-Agent' : "Magic Browser"}) 
                con = urllib.request.urlopen( req )
                webpage = str(con.read().decode("utf-8"))
                soup = bs4.BeautifulSoup(webpage, "lxml")
                article = soup.title.get_text()
                for poccur in soup.find_all('p'):
                    article += " " + poccur.get_text()
                doc = nlp(article)
                if len(doc) >= 100:
                    text_list.append(" ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.pos_ == 'NOUN']))
                link_list = []
                for link in soup.find_all('a'):
                    if not link.img:
                        link_list.append(link['href'])
                link_list = evaluate(url, link_list, visited_list, url_list)
                url_list += link_list
            except:
                print('fail')
                continue
    return text_list
        

def evaluate(url, link_list, visited_list, url_list):
    new_link_list = []
    for x in link_list:
        if url_check(x,url):
            if not x in visited_list and not x in new_link_list and not x in url_list:
                new_link_list.append(x)
    return new_link_list

def url_check(target_url, url):
    if target_url.startswith("http://"):
        target_url = target_url[7:]
    elif target_url.startswith('https://'):
        target_url = target_url[8:]
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith('https://'):
        url = url[8:]
    return target_url.startswith(url)

text_list = explore_website(url, banlist, [])
vec = TfidfVectorizer()
vec.fit(text_list)
print(pandas.DataFrame(vec.transform(text_list).toarray(), columns=sorted(vec.vocabulary_.keys())))
