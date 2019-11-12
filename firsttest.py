import bs4
import urllib.request
import time as t
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas
import json

url = "https://www.votretourdumonde.com"
banlist = ['#comment']
nlp = spacy.load('fr_core_news_md')

def explore_website(url, banlist):
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
                article = {"title": soup.title.get_text(), "text": ""}
                for poccur in soup.find_all('p'):
                    article['text'] += " " + poccur.get_text()
                file = open('./articles/' + alpha_string(url) + '.txt', 'a')
                json.dump(article,file)
                file.write('\n')
                file.close()
                link_list = []
                for link in soup.find_all('a'):
                    if not link.img:
                        link_list.append(link['href'])
                link_list = evaluate(url, link_list, visited_list, url_list)
                url_list += link_list
            except:
                print('fail')
                continue
    return 0
        

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

def alpha_string(string):
    res = ''
    for x in string:
        if x.isalpha():
            res+=x
    return res


def spclabel(text):
    doc = nlp(text)
    return [ent.label_ for ent in doc.ents]

def blogtdidf(blogpath):
    title_list = []
    blog = open(blogpath, 'r')
    lines = blog.readlines()
    tag_list = [[]] * 10
    for x in range(10):
        doc = nlp(json.loads(lines[x+1])['title'] + ' ' + json.loads(lines[x+1])['text'])
        if len(doc) > 400:
            remove_location = " ".join([ent.text for ent in doc.ents if ent.label_ != 'LOC' and ent.text.isalpha()])
            tag_list[x] = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
            doc = nlp(remove_location)
            title_list.append(" ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.isalpha()]))
    passing_tag = []
    for y in tag_list:
        passing_tag.append(" ".join([x for x in y]))
    vec = TfidfVectorizer()
    vec.fit(passing_tag)
    data_frame = pandas.DataFrame(vec.transform(passing_tag).toarray(), columns=sorted(vec.vocabulary_.keys()))
    
#text_list = explore_website(url, banlist)
blogtdidf('./articles/httpswwwvotretourdumondecom.txt')
