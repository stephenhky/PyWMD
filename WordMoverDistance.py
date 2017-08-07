

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import dok_matrix
from gensim.corpora import Dictionary
from gensim.models import keyedvectors
from cvxopt import matrix, solvers
import pulp

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


# too difficult to maintain
def word_mover_distance_cvxopt(first_sent_tokens, second_sent_tokens, wvmodel):
    tokendict, d1vec, d2vec = index_tokens(first_sent_tokens, second_sent_tokens)
    numwords = len(tokendict.token2id)

    c = np.zeros(numwords*numwords)
    for i in range(numwords):
        for j in range(i):
            distance = euclidean(wvmodel[tokendict[i]], wvmodel[tokendict[j]])
            c[singleindexing(numwords, i, j)] = distance
            c[singleindexing(numwords, j, i)] = distance

    # normal constraint
    G = dok_matrix((len(first_sent_tokens)+len(second_sent_tokens), numwords*numwords))
    h = np.zeros(len(first_sent_tokens)+len(second_sent_tokens))
    for i in range(len(first_sent_tokens)):
        for j in range(len(second_sent_tokens)):
            token1idx = tokendict.token2id[first_sent_tokens[i]]
            token2idx = tokendict.token2id[second_sent_tokens[j]]
            G[i, singleindexing(numwords, token1idx, token2idx)] = 1
            h[i] = d1vec[0, token1idx]
    for j in range(len(second_sent_tokens)):
        for i in range(len(first_sent_tokens)):
            token1idx = tokendict.token2id[first_sent_tokens[i]]
            token2idx = tokendict.token2id[second_sent_tokens[j]]
            G[len(first_sent_tokens)+j, singleindexing(numwords, token1idx, token2idx)] = 1
            h[len(first_sent_tokens)+j] = d2vec[0, token2idx]

    # additional constraints
    emptyi = np.where(d1vec.toarray()[0, :]==0)[0]
    emptyj = np.where(d2vec.toarray()[0, :] == 0)[0]
    A = dok_matrix((len(emptyi)*len(second_sent_tokens)+len(emptyj)*len(first_sent_tokens), numwords*numwords))
    b = np.zeros(len(emptyi)*len(second_sent_tokens)+len(emptyj)*len(first_sent_tokens))
    Arowidx = 0
    for i in emptyi:
        for j in range(len(second_sent_tokens)):
            A[Arowidx, singleindexing(numwords, i, j)] = 1
            b[Arowidx] = 0
            Arowidx += 1
    for j in emptyj:
        for i in range(len(first_sent_tokens)):
            A[Arowidx, singleindexing(numwords, i, j)] = 1
            b[Arowidx] = 0
            Arowidx += 1

    # print d1vec.toarray()[0, :]
    # print d2vec.toarray()[0, :]

    print c.shape
    print G.toarray().shape
    print h.shape
    print A.toarray().shape
    print b.shape

    # print tokendict.token2id
    #
    # print matrix(c)
    # print G.toarray()
    # print matrix(h)
    # print A.toarray()
    # print matrix(b)

    sol = solvers.lp(matrix(c), matrix(G.toarray()), matrix(h), matrix(A.toarray()), matrix(b))

    return sol


# use LuLP
def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel):
    pass
# example: tokens1 = ['american', 'president']
#          tokens2 = ['chinese', 'chairman', 'king']