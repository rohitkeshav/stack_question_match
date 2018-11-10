import numpy as np
import pandas as pd

from utils import get_doc_tokens
from scipy.spatial import distance


# TF_IDF on the list of docs
# there definitely a better of doing this
def tf_idf(docs, check_with=None):

    if check_with is not None:
        docs.insert(0, check_with)

    docs_tokens = {idx: get_doc_tokens(doc) for idx, doc in enumerate(docs)}

    # Get document-term matrix
    dtm = pd.DataFrame.from_dict(docs_tokens, orient="index")
    dtm = dtm.fillna(0)

    # Get normalized term frequency (tf) matrix
    tf = dtm.values
    doc_len = tf.sum(axis=1)
    tf = np.divide(tf.T, doc_len).T

    # Get document frequency df
    df = np.where(tf > 0, 1, 0)

    # Get smoothed tf_idf
    smoothed_idf = np.log(np.divide(len(docs) + 1, np.sum(df, axis=0) + 1)) + 1
    smoothed_tf_idf = tf * smoothed_idf

    # calculate cosine distance of every pair of documents
    # convert the distance object into a square matrix form
    # similarity is 1-distance
    similarity = 1 - distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))

    # find top 5 docs similar to first one
    similar_docs = np.argsort(similarity)[:, ::-1][0, 1:4]
    print('similar docs', similar_docs)

    retval = list()
    # Print the top 5 similar reviews
    for idx, doc in enumerate(docs):
        if idx in similar_docs:
            print(docs[idx])
            retval.append(docs[idx])

    return retval
