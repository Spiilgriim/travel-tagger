import bs4
import urllib.request
import time as t
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans
import pandas
import json
import gensim
import random as r
from tqdm import tqdm
url = "https://oiseaurose.com/"
banlist = ['#comment', '#respond', '#profile',
           '?replytocom', '.jpg', '.jpeg', '.png', '.gif']
nlp = spacy.load('fr_core_news_md')


def explore_website(url, banlist):
    i = 0
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
                req = urllib.request.Request(
                    actual_url, headers={'User-Agent': "Magic Browser"})
                con = urllib.request.urlopen(req)
                webpage = str(con.read().decode("utf-8"))
                soup = bs4.BeautifulSoup(webpage, "lxml")
                article = {"title": soup.title.get_text(), "text": "",
                           "link": actual_url}
                for poccur in soup.find_all('p'):
                    article['text'] += " " + poccur.get_text()
                file = open('./articles/' + alpha_string(url) + '.txt', 'a')
                json.dump(article, file)
                file.write('\n')
                file.close()
                link_list = []
                for link in soup.find_all('a'):
                    if link.get("href") and not link.img:
                        link_list.append(link.get('href'))
                link_list = evaluate(url, link_list, visited_list, url_list)
                url_list += link_list

            except:
                print('fail')
                continue

    return 0


def evaluate(url, link_list, visited_list, url_list):
    new_link_list = []
    for x in link_list:
        if url_check(x, url):
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
            res += x
    return res


def spclabel(text):
    doc = nlp(text)
    return [ent.label_ for ent in doc.ents]


def blog_preprocess(blogpath, location=False, clustering=False, random=1):
    r.seed(t.time())
    article_content = []
    blog = open(blogpath, 'r')
    lines = blog.readlines()
    n = len(lines)
    tag_list = []
    link_list = []
    passing_tag = []
    article_list = []
    print('Pre-process' + ' ' + blogpath)
    pbar = tqdm(total=n)
    for x in range(n):
        json_text = json.loads(lines[x])
        doc = nlp(json_text['title'] +
                  ' ' + json_text['text'])
        if len(doc) > 400 and r.random() < random:
            article_list.append(json_text['title'])
            link_list.append(json_text['link'])
            remove_location = " ".join(
                [ent.text for ent in doc.ents if ent.label_ != 'LOC' and ent.text.isalpha()])
            tag_list.append(
                [ent.text for ent in doc.ents if ent.label_ == 'LOC' and ent.text.isalpha()])
            passing_tag.append(" ".join([element for element in tag_list[-1]]))
            doc = nlp(remove_location)
            article_content.append(
                [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.isalpha()])
        pbar.update(1)
    pbar.close()
    if location:
        return article_content
    elif clustering:
        return article_content, article_list
    else:
        return passing_tag, article_list


def serialize_pre_process(blogpaths, location=False, clustering=False, random=[]):
    if len(random) < len(blogpaths):
        random += [1] * len(blogpaths) - len(random)
    if not location:
        article_content_agglo = []
        for index, x in enumerate(blogpaths):
            article_content_agglo += blog_preprocess(x, random[index])
        return article_content_agglo
    else:
        res, article_list_agglo = [], []
        for index, x in enumerate(blogpaths):
            res_temp, article_list_agglo_temp = blog_preprocess(
                x, location=location, clustering=clustering, random=random[index])
            res += res_temp
            article_list_agglo += article_list_agglo_temp
        return res, article_list_agglo


def location_tag(passing_tag, article_list):
    vec = CountVectorizer(binary=False)
    vec.fit(passing_tag)
    res = pandas.DataFrame(vec.transform(passing_tag).toarray())
    maxidx = res.idxmax(axis=1)
    maximum = res.max(axis=1)
    tags = sorted(vec.vocabulary_.keys())
    for i in range(len(res)):
        print(article_list[i], tags[maxidx[i]], maximum[i])


def topic_modelling(article_content, model_name):
    dictionnary = gensim.corpora.dictionary.Dictionary(article_content)
    corpus = [dictionnary.doc2bow(text) for text in article_content]

    lda = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=9, id2word=dictionnary)
    lda.save(model_name + '.gensim')
    topics = lda.print_topics(num_words=4)
    for topic in topics:
        print(topic)


def updateModelWith(other_article_content, model_to_update):
    dictionnary = gensim.corpora.dictionary.Dictionary(other_article_content)
    corpus = [dictionnary.doc2bow(text) for text in other_article_content]

    lda = gensim.models.ldamodel.LdaModel.load(
        model_to_update + '.gensim', mmap='r')
    lda.update(corpus)
    topics = lda.print_topics(num_words=4)
    for topic in topics:
        print(topic)


def clustering(article_content, article_list):
    article_content = " ".join(article_content)
    vec2 = TfidfVectorizer()
    vec2.fit(article_content)
    features = vec2.transform(article_content)

    n_clusters = 9
    cls = MiniBatchKMeans(n_clusters=n_clusters)
    cls.fit(features)
    cls.predict(features)
    clusters = cls.labels_
    for i in range(n_clusters):
        print(i)
        index_list = [j for j, x in enumerate(clusters) if x == i]
        for article in index_list:
            print(article_list[article])
        print('\n\n\n\n\n')


url_list = ['https://www.bons-plans-voyage-new-york.com/', 'https://www.decouvertemonde.com/', 'https://onedayonetravel.com/',
            'https://www.unsacsurledos.com/', 'https://www.madame-oreille.com/', 'https://www.worldelse.com/', 'https://milesandlove.com/', 'https://www.travel-me-happy.com/']

for x in url_list:
    text_list = explore_website(x, banlist)

# updateModelWith(blog_preprocess(
#    './articles/httpswwwvotretourdumondecom.txt', 0.33), 'model2')
# topic_modelling(blog_preprocess(
#   './articles/httpscarnetstraversecom.txt', 0.5), 'model2')

"""
Next to do :
'https://www.gaijinjapan.org/',
'https://lovelivetravel.fr/',
'https://maathiildee.com/',
'https://www.instinct-voyageur.fr/',
'https://www.novo-monde.com/',
'https://cloetclem.fr/',
'https://www.voyagesetc.fr/',
'https://www.lostintheusa.fr/'
"""
