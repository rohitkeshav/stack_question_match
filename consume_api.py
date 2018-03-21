import sys
import html
import time

from settings import SITE
from similiarity import cosine
from utils import store_as_csv, read_from_csv

"""
.csv format - title | user | up-votes | tags
"""

CURRENT_MOD = sys.modules[__name__]


def __question(tag, up_votes):
    data = SITE.fetch('questions/', tagged=tag, sort='votes', min=up_votes, max_pages=1000)

    for question_object in data['items']:
        yield question_object


def parse_and_store():
    ret_val = list()
    n_row = __question('python', 20)

    header_list = ['title', 'tags', 'username', 'creation_date', 'up_votes']

    while True:
        try:
            a_row = next(n_row)

            if a_row['is_answered']:

                ret_val.append(dict(zip(header_list, [html.unescape(a_row['title']), ' '.join(a_row['tags']),
                                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a_row['creation_date'])),
                                        a_row['owner']['display_name'], a_row['score']])))

        except StopIteration:
            break

    store_as_csv(header_list, ret_val)

    return ret_val


def run(ques):
    parse_and_store()
    print(cosine.tf_idf(read_from_csv(), check_with=ques))


input_ques = input('Enter Question? \n')
run(input_ques)
