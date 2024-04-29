import string
import pandas as pd
import numpy as np
import random
import challenge_basic as cb
import train_methods as tm
file_name = "clean_quercus.csv"

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
    #print(data)

    vocabl = list(vocab)

    X_train, t_train = data[:400], vocabl#572 in total
    #X_valid, t_valid = tm.NB_make_bow(data[400:457], vocabl)
    #X_test, t_test = tm.NB_make_bow(data[457:], vocabl)
    X_train1, t_train1 = tm.NB_make_bow(data[:400], vocabl, "1")  # 572 in total
    X_train2, t_train2 = tm.NB_make_bow(data[:400], vocabl, "2")  # 572 in total
    X_train3, t_train3 = tm.NB_make_bow(data[:400], vocabl, "3")  # 572 in total
    t_train = tm.NB_make_train(data[:400], vocabl)
    pi_map1, theta_map1 = tm.naive_bayes_map(X_train1, t_train1, 2, 2)
    pi_map2, theta_map2 = tm.naive_bayes_map(X_train2, t_train2, 2, 2)
    pi_map3, theta_map3 = tm.naive_bayes_map(X_train2, t_train2, 2, 2)
    print("pi_map1",pi_map1, theta_map1)
    print("pi_map2",pi_map2, theta_map2)
    print("pi_map3",pi_map3, theta_map3)
    y_map_train1 = tm.NB_make_prediction(X_train1, pi_map1, theta_map1)
    y_map_train2 = tm.NB_make_prediction(X_train2, pi_map2, theta_map2)
    y_map_train3 = tm.NB_make_prediction(X_train3, pi_map3, theta_map3)
    y_map_train = np.argmax(np.array(list(zip(y_map_train1, y_map_train2, y_map_train3))), axis = 1) + 1
    print("MAP Train Acc:", tm.accuracy(y_map_train, t_train))