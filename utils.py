import csv
import nltk
import time
import random
import string
import requests

from bs4 import BeautifulSoup
from settings import BASE_DIR
from nltk.corpus import stopwords


# tokenize a doc
def get_doc_tokens(doc):
    stop_words = stopwords.words('english')

    words = [token.strip() for token in nltk.word_tokenize(doc.lower()) if
             token.strip() not in stop_words and token.strip() not in string.punctuation]

    token_count = {token: words.count(token) for token in set(words)}

    return token_count


def read_from_csv(filename='/data_set.csv'):
    docs = list()

    with open(BASE_DIR + filename, 'r', encoding='utf-8') as f:
        q_a = csv.DictReader(f)
        for line in q_a:
            docs.append(line['title'])

    return docs


def store_as_csv(columns, lod, file_name='/data_set.csv'):
    with open(BASE_DIR + file_name, 'w', encoding="utf-8") as f:
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
