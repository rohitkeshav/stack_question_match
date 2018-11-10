import csv
import nltk
import time
import random
import string
import requests
import pickle

from bs4 import BeautifulSoup
from settings import BASE_DIR
from nltk.corpus import stopwords

"""
    Useful standalone functions for the project
"""


# tokenize a doc
def get_doc_tokens(doc, j_tokens=False):

    stop_words = stopwords.words('english')
    mt = str.maketrans('', '', string.punctuation)

    words = [token.strip() for token in nltk.word_tokenize(doc.translate(mt).lower()) if
             token.strip() not in stop_words and token.strip() not in string.punctuation]

    if j_tokens:
        return words

    token_count = {token: words.count(token) for token in set(words)}

    return token_count


def read_from_csv(filter_this, filename='/data_set.csv'):
    docs = list()
    d_dict = dict()

    with open(BASE_DIR + filename, 'r', encoding='utf-8') as f:
        q_a = csv.DictReader(f)
        for line in q_a:
            if line['p_lang'] == filter_this:
                docs.append(line['title'])
                d_dict[line['title']] = {'url': line['link']}

    with open('f_data.pickle', 'wb') as handle:
        pickle.dump(d_dict, file=handle, protocol=pickle.HIGHEST_PROTOCOL)

    return docs


def store_as_csv(columns, lod, file_name='/data_set.csv'):
    with open(BASE_DIR + file_name, 'a', encoding="utf-8") as f:
        q_a = csv.DictWriter(f, fieldnames=columns)
        q_a.writeheader()
        q_a.writerows(lod)


# Use later, in case it
# requires to scrape description of a page

def parse_page(url_string):
    r = requests.get(url_string)

    # else stack blocks the scraper
    time.sleep(random.randrange(1, 6))

    if r.status_code != 200:
        return 'Issue with the request'

    soup = BeautifulSoup(r.text, 'html.parser')

    return soup.find('div', {'class': 'post-text'}).get_text()


def tag_plus_title(filename='/data_set.csv'):
    """
    :param filename: file name
    :return: list of title's appended with tags, for better features
    """

    docs = list()

    with open(BASE_DIR + filename, 'r', encoding='utf-8') as f:
        q_a = csv.DictReader(f)

        jargon = ['python', 'python-3.x', 'python-2.x', 'python-2.7']

        for line in q_a:
            docs.append(line['title'] + ' ' + ' '.join([tag for tag in line['tags'].split(' ') if tag not in jargon]))

    return docs
