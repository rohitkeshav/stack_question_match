import numpy as np
import pandas as pd

from utils import get_doc_tokens, tag_plus_title
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# TF_IDF on the list of docs
def tf_idf(docs, check_with=None):
    # Process all documents to get list of token list

    # TODO: apparently not a hack
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

    # Print the top 5 similar reviews
    for idx, doc in enumerate(docs):
        if idx in similar_docs:
            print('top', idx, doc)


def multinomial_nb():
    text_data = tag_plus_title()

    #  Init CountVectorizer
    c_vect = CountVectorizer()

    c_vect.fit(text_data)

    # transform training data into a 'document-term matrix'
    dtm = c_vect.transform(text_data)

    dtm.toarray()

    pd.DataFrame(dtm.toarray(), columns=c_vect.get_feature_names())


def __try():
    features = ['p_lang', 'title', 'p_num']
    stack_data = pd.read_csv('../data_set.csv', header=None, names=features)

    # define X, y

    X = stack_data.title
    y = stack_data.p_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)

    vect = CountVectorizer()

    vect.fit(X_train)

    # transform training data
    X_train_dtm = vect.transform(X_train)
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    nb = MultinomialNB()

    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print(metrics.accuracy_score(y_test, y_pred_class))


__try()
