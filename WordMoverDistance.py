import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import dok_matrix
from gensim.corpora import Dictionary

singleindexing = lambda m, i, j: m*i+j
unpackindexing = lambda m, k: (k/m, k % m)

def word_mover_distance(first_sent_tokens, second_sent_tokens):
    tokendict = Dictionary([first_sent_tokens, second_sent_tokens])
    numwords = len(tokendict.token2id)

    d1vec = dok_matrix((1, numwords))
    for idx, cnt in tokendict.doc2bow(first_sent_tokens):
        d1vec[0, idx] = cnt
    d2vec = dok_matrix((1, numwords))
    for idx, cnt in tokendict.doc2bow(second_sent_tokens):
        d2vec[0, idx] = cnt