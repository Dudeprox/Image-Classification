# This file contains some lab written training programs debugging information
# will be written in content of each method There are currently 3 methods:
# naive bayes, KNN and decision tree, if other methods needs to be added,
# go to this file.
import numpy as np

import re
import string

import pandas as pd
import numpy as np

file_name = "clean_quercus.csv"
import random

random.seed(42)


# Naive Bayes
# Naive Bayes
# Naive Bayes
# NOT YET DEBUGGED

def NB_make_bow(data, vocab, det):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary
        'det' : the positive value
    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])
    t = np.zeros([len(data)])

    # TODO: fill in the appropriate values of X and t
    for i in range(len(data)):
        for j in range(len(vocab)):
            if vocab[j] in data[i][0]:
                X[i][j] = 1
            if data[i][1] == det:
                t[i] = 1
    return X, t

def NB_make_train(data, vocab):
    t = np.zeros([len(data)])
    for i in range(len(data)):
        for j in range(len(vocab)):
            if data[i][1] == "1":
                t[i] = 1
            elif data[i][1] == "2":
                t[i] = 2
            else:
                t[i] = 3
    return t
def naive_bayes_map(X, t, alpha, beta):
    """
    Compute the parameters $pi$ and $theta_{jc}$ that maximizes the log-likelihood
    of the provided data (X, t).

    **Your solution should be vectorized, and contain no loops**

    Parameters:
        `X` - a matrix of bag-of-word features of shape [N, V],
              where N is the number of data points and V is the vocabulary size.
              X[i,j] should be either 0 or 1. Produced by the make_bow() function.
        `t` - a vector of class labels of shape [N], with t[i] being either 0 or 1.
              Produced by the make_bow() function.

    Returns:
        `pi` - a scalar; the MLE estimate of the parameter $\pi = p(c = 1)$
        `pi2` - a scalar; the MLE estimate of the parameter $\pi = p(c = 2)$
        `theta` - a matrix of shape [V, 3], where `theta[j, c]` corresponds to
                  the MLE estimate of the parameter $\theta_{jc} = p(x_j = 1 | c)$
    """
    N, vocab_size = X.shape[0], X.shape[1]
    pi = (np.count_nonzero(t) + alpha - 1) / (N + alpha + beta - 2)
    theta = np.zeros([vocab_size, 2])  # TODO

    # these matrices may be useful (but what do they represent?)
    X_positive = X[t == 1]
    X_negative = X[t == 0]

    theta[:, 1] = (np.count_nonzero(X_positive, axis=0) + alpha - 1) / (X_positive.shape[0] + alpha + beta - 2)
    theta[:, 0] = (np.count_nonzero(X_negative, axis=0) + alpha - 1) / (X_negative.shape[0] + alpha + beta - 2)
    return pi, theta


def NB_make_prediction(X, pi, theta):
    y = np.zeros(X.shape[0])
    log_theta = np.log(theta)

    for i in range(X.shape[0]):
        pos_pred = pi * np.exp(X[i] @ log_theta[:, 1])

        y[i] = pos_pred

    return y

def accuracy(y, t):
    return np.mean(y == t)
# KNN
# KNN
# KNN
# NOT YET DEBUGGED

# Decision Tree
# Decision Tree
# Decision Tree
# NOT YET DEBUGGED

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Clean numerics
    del df["q_quote"]
    del df["q_sell"]
    del df["q_scary"]
    del df["q_dream"]
    del df["q_desktop"]
    del df["q_better"]
    del df["q_remind"]
    del df["q_temperature"]
    del df["user_id"]

    df = df.dropna()
    df = df.reset_index(drop=True)

    data = []

    vocab = set()

    for i, row in df.iterrows():
        line = row[0].replace(u'\xa0', u' ').replace('.', ''). \
            replace(u'\n', u' ').translate(
            str.maketrans('', '', string.punctuation)). \
            strip().lower()

        data.append((line, row[1]))
        vocab.update(line.split())

    data = np.array(data)

    random.shuffle(data)  # Shuffle Data, to mixed labelled examples
    print(data)

    vocabl = list(vocab)

    X_train, t_train = NB_make_bow(data[:1000], vocabl)
    X_valid, t_valid = NB_make_bow(data[1000:], vocabl)

    print(X_train)

    vocab_count_mapping = list(zip(vocabl, np.sum(X_train, axis=0)))
    vocab_count_mapping = sorted(vocab_count_mapping, key=lambda e: e[1], reverse=True)
    for word, cnt in vocab_count_mapping:
        print(word, cnt)
