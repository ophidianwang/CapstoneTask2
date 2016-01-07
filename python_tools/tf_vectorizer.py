#!/usr/bin/python

import math
import json
import pickle
import random
from gensim import models
from gensim import matutils
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import glob
import argparse
import os


def sim_matrix():
    global tmp_text_vec
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # K_clusters = 10
    """
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     min_df=2, stop_words='english',
                                     use_idf=True)
    """

    vectorizer = CountVectorizer(max_df=0.5, max_features=10000, min_df=2,
                                 stop_words='english')  # try different vectorizer

    if not os.path.isdir("categories"):
        print "you need to generate the cuisines files 'categories' folder first"
        return

    text = []
    c_names = []
    cat_list = glob.glob("categories/*")
    cat_size = len(cat_list)
    if cat_size < 1:
        print "you need to generate the cuisines files 'categories' folder first"
        return

    sample_size = min(30, cat_size)
    cat_sample = sorted(random.sample(range(cat_size), sample_size))
    # print (cat_sample)
    count = 0
    for i, item in enumerate(cat_list):
        if i == cat_sample[count]:
            li = item.split('/')
            cuisine_name = li[-1]
            c_names.append(cuisine_name[:-4].replace("_", " "))  # remove ".txt"
            with open(item) as f:
                text.append(f.read().replace("\n", " "))
            count += 1

        if count >= len(cat_sample):
            print "generating cuisine matrix with:", count, "cuisines"
            break

    if len(text) < 1:
        print "the 'categories' folder does not contain any cuisines. Run this program ussing the '--cuisine' option"
    t0 = time()
    print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)

    with open('tf_result/vectorize_temp.txt', 'w') as f:
        f.write("feature vec. size = " + str(len(vectorizer.get_feature_names())))
        f.write("\n==========\n")

        for l, cat_text_vec in enumerate(X.toarray()):
            f.write("cat. #" + str(l) + " : " + str(c_names[l]))
            f.write("\n")
            tmp_text_vec = list(filter(lambda x: x != 0, cat_text_vec))
            f.write("non zero vec. size = " + str(len(tmp_text_vec)))
            f.write("\n")
            f.write(str(cat_text_vec))
            f.write("\n==========\n")

    cuisine_matrix = []  # similarity of topics
    # computing cosine similarity matrix, use word counts as category's vector
    for i, word_freq_a in enumerate(X.toarray()):
        print ("working on cat. #" + str(i) + " : " + str(c_names[i]))
        tmp_word_freq = list(filter(lambda x: x != 0, word_freq_a))
        print("non zero vec. size = " + str(len(tmp_text_vec)))
        sim_vecs = []
        for j, word_freq_b in enumerate(X.toarray()):
            w_sum = 0
            if i <= j:
                norm_a = 0
                norm_b = 0

                for count_b in word_freq_b:
                    norm_b = norm_b + count_b * count_b

                for order_a, count_a in enumerate(word_freq_a):
                    norm_a = norm_a + count_a * count_a
                    for order_b, count_b in enumerate(word_freq_b):
                        if order_a == order_b:
                            w_sum = w_sum + count_a * count_b

                norm_a = math.sqrt(norm_a)
                norm_b = math.sqrt(norm_b)
                denom = float(norm_a * norm_b)
                w_sum = float(w_sum) / denom

            else:
                w_sum = cuisine_matrix[j][i]
            sim_vecs.append(w_sum)
            print ("\tevaluate sim. to cat. #" + str(j) + " : " + str(c_names[j]) + " ; result : " + str(w_sum))

        cuisine_matrix.append(sim_vecs)

    with open('tf_result/cuisine_sim_matrix.csv', 'w') as f:
        for i_list in cuisine_matrix:
            s = ""
            my_max = max(i_list)
            for tt in i_list:
                s = s + str(tt / my_max) + " "
            s = s.strip()
            f.write(",".join(s.split()) + "\n")  # should the list be converted to m

    with open('tf_result/cuisine_indices.txt', 'w') as f:
        f.write("\n".join(c_names))

    # create json file for d3.js
    output = []
    cursor = 0
    while len(c_names) > cursor:
        single_matrix = []
        i_list = cuisine_matrix[cursor]
        my_max = max(i_list)
        for tt in i_list:
            single_matrix.append(tt / my_max)

        single = {}
        single["category"] = c_names[cursor]
        single["sim_matrix"] = single_matrix

        output.append(single)
        cursor += 1

    with open('tf_result/result.json', 'w') as f:
        f.write(json.dumps(output))


if __name__ == "__main__":
    sim_matrix()
