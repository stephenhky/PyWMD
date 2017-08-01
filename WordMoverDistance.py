
from itertools import product

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import dok_matrix
from gensim.corpora import Dictionary
from gensim.models import keyedvectors
from cvxopt import matrix, solvers

singleindexing = lambda m, i, j: m*i+j
unpackindexing = lambda m, k: (k/m, k % m)


def index_tokens(first_sent_tokens, second_sent_tokens):
    tokendict = Dictionary([first_sent_tokens, second_sent_tokens])
    numwords = len(tokendict.token2id)

    d1vec = dok_matrix((1, numwords))
    for idx, cnt in tokendict.doc2bow(first_sent_tokens):
        d1vec[0, idx] = cnt
    d2vec = dok_matrix((1, numwords))
    for idx, cnt in tokendict.doc2bow(second_sent_tokens):
        d2vec[0, idx] = cnt

    d1vec /= np.sum(d1vec)
    d2vec /= np.sum(d2vec)

    return tokendict, d1vec, d2vec


def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel):
    tokendict, d1vec, d2vec = index_tokens(first_sent_tokens, second_sent_tokens)
    numwords = len(tokendict.token2id)

    c = np.zeros(numwords*numwords)
    for i, j in product(range(numwords), range(numwords)):
        c[singleindexing(numwords, i, j)] = euclidean(wvmodel[tokendict[i]], wvmodel[tokendict[j]])

    G = dok_matrix((numwords*2, numwords*numwords))
    h = np.zeros(numwords*2)
    for i in range(numwords):
        for j in range(numwords):
            G[i, singleindexing(numwords, i, j)] = 1
            h[i] = d1vec[0, i]
    for j in range(numwords):
        for i in range(numwords):
            G[numwords+j, singleindexing(numwords, i, j)] = 1
            h[numwords+j] = d2vec[0, j]

    print c.shape
    print G.toarray().shape
    print h.shape
    sol = solvers.lp(matrix(c), matrix(G.toarray()), matrix(h))

    return sol

# example: tokens1 = ['american', 'president']
#          tokens2 = ['chinese', 'chairman', 'king']