import numpy as np

from utils import get_doc_tokens
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


euc = []
lis = []

doc = list()
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(doc).todense()
quest = vectorizer.fit_transform(ques).todense()


for f in features:
    lis = euclidean_distances(features[2], f)
    euc.append(lis[0][0])
print(euc)
index = [i for i in np.argsort(euc)[:3]]
index = index[1:]

for o in index:
    print(doc[o])

