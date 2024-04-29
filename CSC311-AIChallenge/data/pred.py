# basic python imports are permitted
import sys
import csv
import random
# numpy and pandas are also permitted
import string
import pandas as pd
import numpy as np
import random
import train_methods as tm

pi_map1, theta_map1 = 0, 0
pi_map2, theta_map2 = 0, 0
pi_map3, theta_map3 = 0, 0
vocabl = 0
# numpy and pandas are also permitted
import numpy
import pandas
def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """

    # randomly choose between the three choices: image 1, 2, vs 3.
    y = random.choice([1, 2, 3])

    # return the prediction
    return y

if __name__ == "__main__":
    global pi_map1, theta_map1, pi_map2, theta_map2, pi_map3, theta_map3
    # check if the argument <test_data.csv> is provided
    if len(sys.argv) < 2:
        print("""
Usage:
    python example_pred.py <test_data.csv>

As a first example, try running `python example_pred.py example_test_set.csv`
""")
        exit()

    # store the name of the file containing the test data
    filename = sys.argv[-1]

    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data1 = csv.DictReader(open(filename))
    file_name = "clean_quercus.csv"
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
    # print(data)

    vocabl = list(vocab)

    X_train, t_train = data[:400], vocabl  # 572 in total
    # X_valid, t_valid = tm.NB_make_bow(data[400:457], vocabl)
    # X_test, t_test = tm.NB_make_bow(data[457:], vocabl)
    X_train1, t_train1 = tm.NB_make_bow(data[:400], vocabl, "1")  # 572 in total
    X_train2, t_train2 = tm.NB_make_bow(data[:400], vocabl, "2")  # 572 in total
    X_train3, t_train3 = tm.NB_make_bow(data[:400], vocabl, "3")  # 572 in total
    t_train = tm.NB_make_train(data[:400], vocabl)
    pi_map1, theta_map1 = tm.naive_bayes_map(X_train1, t_train1, 2, 2)
    pi_map2, theta_map2 = tm.naive_bayes_map(X_train2, t_train2, 2, 2)
    pi_map3, theta_map3 = tm.naive_bayes_map(X_train2, t_train2, 2, 2)
    for test_example in data1:
        # obtain a prediction for this test example
        pred = predict(test_example)

        # print the prediction to stdout
        print(pred)