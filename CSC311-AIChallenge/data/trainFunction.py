import random

import numpy as np
import re
import string

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import argmax

import challenge_basic as cb
'''
This file contains and trains on 3 seperate models:
1 KNN, 2 naive bayes
For each data, fetch data from the data file.
input to the the model and then produce the model
then, we are able to predict using different models and just use the highest vote between them.
'''
file_name = "clean_quercus.csv"
def dist_all(v, X):
    """
    Compute the squared Euclidean distance between an image `v` (vector) and the
    images in the data matrix `X`.

    Parameters:
        `v` - a numpy array (vector) representing an MNIST image, shape (784,)
        `X` - a data matrix representing a set of MNIST image, shape (N, 784)

    Returns: a vector of squared Euclidean distances between `v` and each image in `X`,
             shape (N,)
    """

    diff = X - v
    sqdiff = diff ** 2
    sumval = None
    if len(np.shape(sqdiff)) == 1:
      sumval= np.sum(sqdiff)
    else:
      sumval = np.sum(sqdiff, axis=1)
    return sumval

def predict_knn(v, X_train, t_train, k=1):
    """
    Returns a prediction using the k-NN

    Parameters:
        `v` - a numpy array (vector) representing an MNIST image, shape (784,)
        `X_train` - a data matrix representing a set of MNIST image, shape (N, 784)
        `t_train` - a vector of ground-truth labels, shape (N,)
        `k` - a positive integer 1 < k <= N, describing the number of closest images
              to consider as part of the knn algorithm

    Returns:
             returned indicies, then `X[i]` is one of the k closets images to `v`
             in the data set `X`
    """
    dists = dist_all(v, X_train)
    indices = np.argsort(dists)[0:k]
    ts = t_train[np.array(indices)]
    prediction = np.unique(ts, return_index=True, return_inverse=False, return_counts=True)[1:] #correct
    prediction = ts[prediction[0][np.argmax(prediction[:][1])]]
    return prediction

def compute_accuracy(X_new, t_new, X_train, t_train, k=1):
    """
    Returns the accuracy (proportion of correct predictions) on the data set
    `X_new` and ground truth `t_new`.

    Parameters:
        `X_new` - a data matrix representing MNIST images that we would like to
                  make predictions for, shape (N', 784)
        `t_new` - a data matrix representing ground truth labels for images in X_new,
                  shape (N',)
        `X_train` - a data matrix representing a set of MNIST image in the training set,
                    shape (N, 784)
        `t_train` - a vector of ground-truth labels for images in X_train,
                    shape (N,)
        `k` - a positive integer 1 < k <= N, describing the number of closest images
              to consider as part of the knn algorithm

    Returns: the proportion of correct predictions (between 0 and 1)
    """

    num_predictions = 0
    num_correct = 0
    for i in range(X_new.shape[0]): # iterate over each image index in X_new
        v = X_new[i] # image vector
        t = t_new[i] # prediction target

        y = predict_knn(v, X_train, t_train, k)
        num_correct += (y==t)

        num_predictions += 1

    return num_correct / num_predictions


if __name__ == "__main__":
    dfor = pd.read_csv(file_name)
    df1 = dfor.copy()
    # Clean others until we contain only q_temperature
    del df1["q_quote"]
    del df1["q_sell"]
    del df1["q_scary"]
    del df1["q_dream"]
    del df1["q_desktop"]
    del df1["q_better"]
    del df1["q_remind"]
    del df1["q_story"]
    del df1["user_id"]

    df1 = df1.dropna()
    df1 = df1.reset_index(drop=True)
    data = np.array(df1)
    random.shuffle(data)
    print(data)
    print(df1.size) #Total of 583 valid data
    D1_train = data[:466]
    D1_valid = data[466:524]
    D1_test = data[524:]
    x_vectors =[cb.to_numeric(x) for [x, t] in D1_train]
    X_train = np.stack(x_vectors)
    targets = [t for [x, t] in D1_train]
    t_train = np.array(targets)
    X_valid = np.stack([cb.to_numeric(x) for (x, t) in D1_valid])
    t_valid = np.array([t for (x, t) in D1_valid])
    X_test = np.stack([cb.to_numeric(x) for (x, t) in D1_test])
    t_test = np.array([t for (x, t) in D1_test])
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    epsilon = 0.0001
    X_train_norm = (X_train - X_mean) / (X_std + epsilon)
    X_valid_norm = (X_valid - np.mean(X_valid, axis=0)) / (np.std(X_valid, axis=0) + epsilon)
    X_test_norm = (X_test - np.mean(X_test, axis=0)) / (np.std(X_test, axis=0) + epsilon)
    print(x_vectors)
    #print("X_train", X_train)
    prediction = predict_knn(X_valid[0], X_train, t_train, k=5)

    print(prediction, t_valid[0])
    valid_acc_norm = []
    for k in range(1, 200):
        acc = compute_accuracy(X_valid_norm, t_valid, X_train=X_train_norm, t_train=t_train, k=k)  # TODO
        valid_acc_norm.append(acc)
    print(argmax(valid_acc_norm), max(valid_acc_norm))
    plt.title("Validation Accuracy for a Normalized kNN model")
    plt.plot(range(1, 200), valid_acc_norm)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()