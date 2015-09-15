#!/usr/bin/python

import math
import json
import pickle
import random
from gensim import models
from gensim import matutils
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from time import time
import glob
import os
import numpy
from numpy import zeros

def runKMeans(K_cluster, cluster_input):
    #clustering by topic-probability vector of each category
    itr = 1000
    t0 = time()
    km = KMeans(n_init=30,n_clusters=K_cluster,n_jobs=2,max_iter=itr)
    km.fit(cluster_input)
    print("done in %0.3fs" % (time() - t0))

    with open( 'result/kmeans_cluster_' + str(K_cluster) + '.txt', 'w') as f:
        f.write("cluster_centers\n")
        f.write(str( km.cluster_centers_))
        f.write("\n==========\n")
        f.write("labels (sequence of cluster # which input belongs to )\n")
        f.write(str( km.labels_))
        f.write("\n==========\n")
        f.write("inertia\n")
        f.write(str( km.inertia_))
        f.write("\n==========\n")

    return km.labels_

def runBrich(K_cluster, cluster_input):
    #clustering by topic-probability vector of each category
    t0 = time()
    bri = Birch(n_clusters=K_cluster)
    bri.fit(cluster_input)
    print("done in %0.3fs" % (time() - t0))

    with open( 'result/brich_cluster_' + str(K_cluster) + '.txt', 'w') as f:
        f.write("cluster_centers\n")
        f.write(str( bri.subcluster_centers_))
        f.write("\n==========\n")
        f.write("labels (sequence of cluster # which input belongs to )\n")
        f.write(str( bri.labels_))
        f.write("\n==========\n")
        f.write("inertia\n")
        f.write(str( bri.subcluster_labels_ ))
        f.write("\n==========\n")

    return bri.labels_

def sortOutput(labels, c_names, doc_topics, algorithm):
    #sort doc_topics by their label and make output files

    K_cluster = max(labels)+1

    sorted_c_names = []
    sorted_labels = []
    sorted_doc_topics = []
    sorting_label = 0

    while (sorting_label < K_cluster):
        for s, doc_s in enumerate(doc_topics):
            if (labels[s] == sorting_label):
                sorted_c_names.append( c_names[s] )
                sorted_labels.append( sorting_label )
                sorted_doc_topics.append( doc_s )

        sorting_label = sorting_label + 1
    
    with open( 'result/sorted_topic_distribution_' + algorithm + str(K_cluster) + '.txt', 'w') as f:
        for k, doc_c in enumerate(sorted_doc_topics):
            f.write("#" + str(k) + " " + sorted_c_names[k])
            f.write("\n")
            f.write(str(doc_c))
            f.write("\n")

            doc_c_array = zeros(100)

            for (index, value) in doc_c:
                doc_c_array[index] = value

            f.write(str(doc_c_array))
            f.write("\n")

    # computing cosine similarity matrix
    cuisine_matrix = [] #similarity of topics

    for i, doc_a in enumerate(sorted_doc_topics):
        #print (i)
        sim_vecs = []
        for j , doc_b in enumerate(sorted_doc_topics):
            w_sum = 0
            if ( i <= j ):
                norm_a = 0
                norm_b = 0
                
                for (my_topic_b, weight_b) in doc_b:
                    norm_b = norm_b + weight_b*weight_b

                for (my_topic_a, weight_a) in doc_a:
                    norm_a = norm_a + weight_a*weight_a
                    for (my_topic_b, weight_b) in doc_b:
                        if ( my_topic_a == my_topic_b ):
                            w_sum = w_sum + weight_a*weight_b

                norm_a = math.sqrt(norm_a)
                norm_b = math.sqrt(norm_b)
                denom = (float) (norm_a * norm_b)
                if denom < 0.0001:
                    w_sum = 0
                else:
                    w_sum = w_sum/(denom)
            else:
                w_sum = cuisine_matrix[j][i]
            sim_vecs.append(w_sum)

        cuisine_matrix.append(sim_vecs)

    """
    with open( 'result/cuisine_sim_matrix.csv', 'w') as f:
        for i_list in cuisine_matrix:
            s = ""
            my_max = max(i_list)
            for tt in i_list:
                s = s+str(tt/my_max) + " "
            s = s.strip()
            f.write(",".join(s.split())+"\n") #should the list be converted to m
    
    with open('result/cuisine_indices.txt', 'w') as f:
        #f.write( "\n".join(c_names))
        f.write( "\n".join(sorted_c_names))
    """

    #create json file for d3.js
    output = []
    cursor = 0
    while (cursor<len(sorted_c_names)):
        single_matrix = []
        i_list = cuisine_matrix[cursor]
        my_max = max(i_list)
        for tt in i_list:
            single_matrix.append(tt/my_max)

        single = {}
        single["category"] = sorted_c_names[cursor]
        single["cluster"] = sorted_labels[cursor]
        single["sim_matrix"] = single_matrix

        output.append(single)
        cursor = cursor+1

    with open('result/tfidf_lda_' + algorithm + "_" + str(K_cluster) + '.json', 'w') as f:
        f.write( json.dumps(output) )

    return

def sim_matrix():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     min_df=2, stop_words='english',
                                     use_idf=True)
        
    if not os.path.isdir("categories"):
        print "you need to generate the cuisines files 'categories' folder first"
        return
    
    text = []
    c_names = []
    cat_list = glob.glob ("categories/*")
    cat_size = len(cat_list)
    if cat_size < 1:
        print "you need to generate the cuisines files 'categories' folder first"
        return
    
    sample_size = min(30, cat_size)
    cat_sample = sorted( random.sample(range(cat_size), sample_size) )
    
    count = 0
    for i, item in enumerate(cat_list):
        if i == cat_sample[count]:
            li =  item.split('/')
            cuisine_name = li[-1]
            c_names.append(cuisine_name[:-4].replace("_"," "))
            with open ( item ) as f:
                text.append(f.read().replace("\n", " "))
            count = count + 1
        
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

    """DEBUG
    with open( 'result/vectorize_temp.txt', 'w') as f:
        f.write( "feature vec. size = " + str( len( vectorizer.get_feature_names() ) ))
        f.write("\n")
        #f.write("X type" + str( type( X ) ))
        #f.write("\n==========\n")

        for l,cat_text_vec in enumerate(X.toarray()):
            f.write( "cat. #" + str(l) + " : " + str(c_names[l]) )
            f.write("\n")
            tmp_text_vec = list(filter(lambda x: x!= 0, cat_text_vec))
            f.write( "non zero vec. size = " + str( len(tmp_text_vec) ))
            f.write("\n")
            #f.write( str(type(cat_text_vec)) )
            #f.write("\n")
            f.write( str(cat_text_vec) )
            f.write("\n==========\n")
    """

    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=100)

    doc_topics = lda.get_document_topics(corpus)

    cluster_input_list = []

    with open( 'result/topic_distribution.txt', 'w') as f:
        for k, doc_c in enumerate(doc_topics):
            f.write("#" + str(k) + " " + c_names[k])
            f.write("\n")
            f.write(str(doc_c))
            f.write("\n")

            doc_c_array = zeros(100)

            for (index, value) in doc_c:
                doc_c_array[index] = value

            f.write(str(doc_c_array))
            f.write("\n")

            cluster_input_list.append(doc_c_array)

            f.write("==========\n")

        cluster_input = numpy.array(cluster_input_list)
        f.write(str(cluster_input))
        f.write("\n")

    #sort doc_topics by cluster label
    kmeans_labels_16 = runKMeans(16, cluster_input)
    sortOutput(kmeans_labels_16, c_names, doc_topics, 'KMeans')

    kmeans_labels_13 = runKMeans(13, cluster_input)
    sortOutput(kmeans_labels_13, c_names, doc_topics, 'KMeans')

    kmeans_labels_10 = runKMeans(10, cluster_input)
    sortOutput(kmeans_labels_10, c_names, doc_topics, 'KMeans')

    kmeans_labels_7 = runKMeans(7, cluster_input)
    sortOutput(kmeans_labels_7, c_names, doc_topics, 'KMeans')

    brich_labels_13 = runBrich(13, cluster_input)
    sortOutput(brich_labels_13, c_names, doc_topics, 'Brich')

    brich_labels_10 = runBrich(10, cluster_input)
    sortOutput(brich_labels_10, c_names, doc_topics, 'Brich')

    brich_labels_7 = runBrich(7, cluster_input)
    sortOutput(brich_labels_7, c_names, doc_topics, 'Brich')

    """
    brich_labels_None = runBrich(None, cluster_input)
    sortOutput(brich_labels_None, c_names, doc_topics, 'Brich')
    """

if __name__=="__main__":
    sim_matrix()
