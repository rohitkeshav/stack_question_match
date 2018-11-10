import html
import time
import random
import argparse
import pandas as pd
from similiarity import cosine
from classi import Linear_SVC

from settings import SITE
from utils import store_as_csv, read_from_csv
import pickle
import webbrowser

"""
.csv format - title | user | up-votes | tags
"""


def __question(tag):
    """
    :return: question object with params such as title, description etc.
    """

    params = ('fromdate', 'todate')
    set_to = [(1262347200, 1293796800), (1293883200, 1325332800),
              (1325419200, 1357041600), (1362225600, 1393588800),
              (1393675200, 1425211200), (1427889600, 1459512000),
              (1459598400, 1491134400), (1491220800, 1519560000),
              (1230811200, 1262260800)]

    data = list()

    for st in set_to:
        data.extend(SITE.fetch('questions/', tagged=tag, sort='votes', max_pages=1000,
                               **dict(zip(params, st))).get('items', []))

        # or your IP gets banned
        time.sleep(random.randrange(0, 10))

    for question_object in data:
        yield question_object


# Parse API and store in CSV
def parse_and_store(tag):
    """
    stores the data in a csv as well
    :param tag: primary filters - ['python', 'c++', 'c', 'java', 'javascript', 'bash']
    :return: list of dictionaries, as per the tag
    """
    ret_val = list()
    n_row = __question(tag)

    header_list = ['title', 'tags', 'creation_date', 'username', 'up_votes', 'link', 'p_lang']

    while True:
        try:
            a_row = next(n_row)

            if a_row['is_answered']:

                ret_val.append(dict(zip(header_list, [html.unescape(a_row['title']), ' '.join(a_row['tags']),
                                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a_row['creation_date'])),
                                        a_row['owner'].get('display_name', 'Unknown User'), a_row['score'], a_row['link'], tag])))

        except StopIteration:
            break

    # stores it into data_set.csv
    store_as_csv(header_list, ret_val)

    return ret_val


def scrape():
    # get these tags only
    language_list = ['python', 'c++', 'c', 'java', 'javascript', 'bash']

    for ll in language_list:
        parse_and_store(ll)


def run(ques):
    """
        Scraping data
    """

    scrape()
    df = pd.read_csv("data_set.csv")
    pred_tag = Linear_SVC(df, ques)

    retval = cosine.tf_idf(read_from_csv(filter_this=pred_tag), check_with=ques)

    with open('f_data.pickle', 'rb') as handle:
        b = pickle.load(handle)

    for i in retval:
        webbrowser.open_new_tab(b[i]['url'])


"""
    Sample Usage
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()

    # run(ques='what is abstract classes in Java?')
    run(ques=args.query)
