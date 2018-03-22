import html
import time

from settings import SITE
from similiarity import cosine
from utils import store_as_csv, read_from_csv

"""
.csv format - title | user | up-votes | tags
"""


def __question(tag, up_votes):

    params = ('fromdate', 'todate')
    set_to = [(1262347200, 1293796800), (1293883200, 1325332800),
              (1325419200, 1357041600), (1362225600, 1393588800),
              (1393675200, 1425211200), (1427889600, 1459512000),
              (1459598400, 1491134400), (1491220800, 1519560000)]

    data = list()

    for st in set_to:
        # time.sleep(10)
        data.extend(SITE.fetch('questions/', tagged=tag, sort='votes', min=up_votes, max_pages=1000,
                               **dict(zip(params, st))).get('items', []))

    for question_object in data:
        yield question_object


# Parse API and store in CSV
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
