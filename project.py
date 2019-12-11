#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 09:48:53 2019

@author: Julie Courgibet
"""

# imports
from scipy.io import loadmat
import numpy as np
import collections


def kppv(apprent, classe_origine, k, x):
    # Shape of apprent
    nbFeatures = apprent.shape[0]
    nbIndividus = apprent.shape[1]
    nbToClass = x.shape[1]

    # Initialisation labels
    labels = []

    # Calcul matrice distances
    for ind in range(nbToClass):
        # Features from individual ind
        indFeature = np.zeros([2,1])
        for features in range(nbFeatures):
            indFeature[features, 0] = x[features, ind]
        
        # Initialisation matrice distance
        distance = np.zeros([1, nbIndividus])
        
        # Calcul distance pour tous les individus apprent
        for ind2 in range(nbIndividus):
            # Features from individual ind2
            indFeature2 = np.zeros([2,1])
            for features in range(nbFeatures):
                indFeature2[features, 0] = apprent[features, ind2]

            # Distance calculation
            mat = indFeature - indFeature2
            matTranspose = np.transpose(mat)
            dist = np.dot(matTranspose, mat)
            distance[0, ind2] = dist
        
        # k closest neighbors
        closestDist = np.argsort(distance)
        labelsClosest = []
        for elt in closestDist[0, :k]:
            labelsClosest.append(classe_origine[0][elt])
        counter = collections.Counter(labelsClosest)
        value = counter.most_common(1)
        labels.append(value[0][0])
        
    return labels


def calculScore(listeLabels):
    score = 0
    for i in range(50):
        if listeLabels[i] == 1:
            score += 1
    for i in range(50, 100):
        if listeLabels[i] == 2:
            score += 1
    for i in range(100, 150):
        if listeLabels[i] == 3:
            score += 1
    return score


if __name__ == "__main__":
    data = loadmat("p1_data1.mat")
    labels_kppv = kppv(data["x"], data["clasapp"], 5, data["test"])
    score_kppv = calculScore(labels_kppv)
    print(score_kppv/len(labels_kppv)*100)

