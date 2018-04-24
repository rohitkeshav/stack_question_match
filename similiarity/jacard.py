from __future__ import division
import string
import math
import numpy as np
from utils import get_doc_tokens


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))

    return len(intersection) / len(union)


def run(docs):
    t_docs = [get_doc_tokens(doc, j_tokens=True) for doc in docs]

    sim = list()

    for td in t_docs:
        sim.append(jaccard_similarity(t_docs[4], td))

    print(sim)

    index = [i for i in np.argsort(sim)[-3:]]
    index = index[:-1]
    index.reverse()

    for idx in index:
        print(t_docs[idx])
