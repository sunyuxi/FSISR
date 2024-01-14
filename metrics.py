# -*- coding: utf-8 -*-
#

from __future__ import division, print_function
import numpy as np
import sys
import time


#######################################################
# Evaluate Critiron
#######################################################


def mean_average_precision(database_hash, test_hash, database_labels, test_labels, R, T): 
    start = time.time()

    # binary the hash code
    #R = args.R
    #T = args.T
    database_hash[database_hash<T] = -1
    database_hash[database_hash>=T] = 1
    test_hash[test_hash<T] = -1
    test_hash[test_hash>=T] = 1

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)  
    ids = np.argsort(-sim, axis=0)  
    
    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i, :]  # test labels
        if np.sum(label) == 0:
            assert(False)

        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / R)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    end = time.time()
    print('dot time: ' + str(end - start), flush=True)
    
    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx

# sim: size(database_count, test_count)
def mean_average_precision_with_sim(sim, database_labels, test_labels, R):  # R = 1000
    start = time.time()
    query_num = test_labels.shape[0]  # total number for testing
    ids = np.argsort(-sim, axis=0)  
    
    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i, :]  # test labels
        if np.sum(label) == 0:
            assert(False)

        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / R)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    end = time.time()
    print('dot time: ' + str(end - start), flush=True)
    
    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx