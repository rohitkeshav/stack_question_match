from __future__ import division
import string
import math
import numpy as np
 
tokenize = lambda doc: doc.lower().split(" ")
sim =[]
 
document_0 = "China has a strong economy that is growing at a rapid pace."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"



all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

tokenized_documents = [tokenize(d) for d in all_documents] # tokenized docs
all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])

for i in tokenized_documents:
    a = jaccard_similarity(tokenized_documents[4],i)
    sim.append(a)
print(sim)
index = [i for i in np.argsort(sim)[-3:]]
index = index[:-1]
index.reverse()
for o in index:
    print(all_documents[o])
    


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union) 

