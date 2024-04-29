#This file contains some lab written training programs
#debugging information will be written in content of each method
#There are currently 3 methods: naive bayes, KNN and decision tree, if other methods needs to be added, go to this file.
import numpy as np

#Naive Bayes
#Naive Bayes
#Naive Bayes
#NOT YET DEBUGGED
def NB_make_bow(data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])

    # TODO: fill in the appropriate values of X and t
    for each_phrase in range(len(vocab)):
      for each_sentense in range(len(data)):
        if vocab[each_phrase] in data[each_sentense][0]:
          X[each_sentense][each_phrase] = 1
    return X

def NB_naive_bayes_mle(X, t):
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
        `theta` - a matrix of shape [V, 2], where `theta[j, c]` corresponds to
                  the MLE estimate of the parameter $\theta_{jc} = p(x_j = 1 | c)$
    """
    N, vocab_size = X.shape[0], X.shape[1]
    pi = 0 # TODO
    theta = np.zeros([vocab_size, 2]) # TODO

    # these matrices may be useful (but what do they represent?)
    X_positive = X[t == 1]
    X_negative = X[t == 0]
    pi = X_positive.shape[0] / N
    theta[:, 1] = np.sum(X_positive, axis = 0) / X_positive.shape[0]
    theta[:, 0] = np.sum(X_negative, axis = 0) / X_negative.shape[0]
    return pi, theta

def NB_make_prediction(X, pi, theta):
    #print((X * theta[:, 0])[0])
    #print(np.sum(X * np.log(theta[:, 0]), axis = 1))
    #y0_tmp = X * np.log(theta[:, 0], where=(theta[:, 0] != 0))
    #y1_tmp = X * np.log(theta[:, 1], where=(theta[:, 1] != 0))
    y0_tmp = X * np.log(theta[:, 0])
    y1_tmp = X * np.log(theta[:, 1])
    y0 = np.exp(np.log(1 - pi) + np.sum(y0_tmp, axis = 1))
    y1 = np.exp(np.log(pi) + np.sum(y1_tmp, axis = 1))
    #print(y0)
    #print(y1)
    #print("y0", y0)
    #print("y1", y1)
    y = y1 > y0
    #print(y[y == False])
    return y


#KNN
#KNN
#KNN
#NOT YET DEBUGGED

#Decision Tree
#Decision Tree
#Decision Tree
#NOT YET DEBUGGED