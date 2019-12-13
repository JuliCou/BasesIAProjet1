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
from math import pi, sqrt, exp, pow


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


def entrainementBayes(apprent, classe_origine):
    # Shape of apprent
    nbFeatures = apprent.shape[0]
    nbIndividus = apprent.shape[1]
    k = len(np.unique(classe_origine))

    # Initialisation des paramètres
    # moyenne
    m = np.zeros([nbFeatures, k])
    # Matrice des variances / covariance pour chaque classe
    sigma = np.zeros([nbFeatures, nbFeatures, k])
    # Proba
    p = np.zeros(k)

    # Calcul des paramètres
    # Pour chaque classe
    for c in range(k):
        # Obtention de la sous-matrice pour la classe c
        nb = np.count_nonzero(classe_origine == c+1)
        submatrix = np.zeros([nbFeatures, nb])
        for f in range(nbFeatures):
            n = 0
            for i in range(nbIndividus):
                if classe_origine[i] == c+1:
                    submatrix[f, n] = apprent[f, i]
                    n += 1

        # Calcul de la moyenne
        for f in range(nbFeatures):
            m[f, c] = np.mean(submatrix[f, :])

        # Calcul de la matrice de covariance
        sigma[:, :, c] = np.cov(submatrix)

        # Probabilités
        p[c] = nb/nbIndividus

    return m, sigma, p


def bayes(m, sigma, p, x):
    # Constantes
    nbFeatures = m.shape[0]
    nbClasses = m.shape[1]
    nbToClass = x.shape[1]
    
    # Labels to return
    labels = []

    # Calcul probabilités pour chaque
    # point à prédire
    for pt in range(nbToClass):
        # Calcul des probas
        proba = np.zeros([nbFeatures, nbClasses])
        probaF = np.zeros(nbClasses)
        for c in range(nbClasses):
            probaF[c] = p[c]
            for f in range(nbFeatures):
                proba[f, c] = 1/sqrt(2*pi)*exp(-pow(x[f, pt]-m[f, c], 2)/(2*pow(sigma[f, f, c], 2)))
                probaF[c] *= proba[f, c]

        # Maximum value
        labels.append(np.argmax(probaF)+1)

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

    # K plus proches voisins
    labels_kppv = kppv(data["x"], data["clasapp"], 5, data["test"])
    score_kppv = calculScore(labels_kppv)
    print(score_kppv/len(labels_kppv)*100)

    # Naive Bayes
    m, sigma, p = entrainementBayes(data["x"], data["clasapp"][0])
    labels_bayes = bayes(m, sigma, p, data["test"])
    score_bayes = calculScore(labels_bayes)
    print(score_bayes/len(labels_kppv)*100)

