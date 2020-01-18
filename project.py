#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/12/2019
Last modified on 15/01/2020

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
            labelsClosest.append(classe_origine[elt])
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
        nb = classe_origine.count(c+1)
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
                proba[f, c] = 1/sqrt(2*pi*sigma[f, f, c])*exp(-pow(x[f, pt]-m[f, c], 2)/(2*sigma[f, f, c]))
                probaF[c] *= proba[f, c]

        # Maximum value
        labels.append(np.argmax(probaF)+1)

    return labels


def calculScore(listeLabels, vraieValeur):
    score = 0
    for i in range(len(listeLabels)):
        if listeLabels[i] == vraieValeur[i]:
            score += 1
    return score


if __name__ == "__main__":

    ## 2.1. Comparaison standard
    data_1 = loadmat("p1_data1.mat")

    # K plus proches voisins
    print("Partie 2.1.")
    classes = [1] * 50
    classes += [2] * 50
    classes += [3] * 50
    for i in range(8):
        labels_kppv = kppv(data_1["test"], classes, 2*i+1, data_1["x"])
        score_kppv = calculScore(labels_kppv, data_1["clasapp"][0])
        print("k : ", 2*i+1, "score : ", score_kppv/len(labels_kppv)*100)

    # Naive Bayes
    m, sigma, p = entrainementBayes(data_1["test"], classes)
    labels_bayes = bayes(m, sigma, p, data_1["x"])
    score_bayes = calculScore(labels_bayes, data_1["clasapp"][0])
    print("Naive Bayes : ", score_bayes/len(labels_bayes)*100)

    ## 2.2. Absence de professeur
    print("Partie 2.2.")
    data_2 = loadmat("p1_data2.mat")

    # K plus proches voisins
    for i in range(8):
        labels_kppv_2 = kppv(data_2["test"], data_2["orig"][0], 2*i+1, data_2["x"])
        score_kppv_2 = calculScore(labels_kppv_2, data_2["clasapp"][0])
        print("k : ", 2*i+1, "score : ", score_kppv_2/len(labels_kppv_2)*100)

    # Naive Bayes
    m, sigma, p = entrainementBayes(data_2["test"], list(data_2["orig"][0]))
    labels_bayes_2 = bayes(m, sigma, p, data_2["x"])
    score_bayes_2 = calculScore(labels_bayes_2, data_2["clasapp"][0])
    print("Naive Bayes : ", score_bayes_2/len(labels_bayes_2)*100)

    ## 2.3. Influence de la taille de l'ensemble d'apprentissage
    print("Partie 2.3.")
    data_3a = loadmat("p1_data3a.mat")

    # K plus proches voisins
    classes = [1] * 20
    classes += [2] * 20
    classes += [3] * 20
    for i in range(8):
        labels_kppv_3a = kppv(data_3a["test"], classes, 2*i+1, data_3a["x"])
        score_kppv_3a = calculScore(labels_kppv_3a, data_3a["clasapp"][0])
        print("k : ", 2*i+1, "score : ", score_kppv_3a/len(labels_kppv_3a)*100)

    # Naive Bayes
    m, sigma, p = entrainementBayes(data_3a["test"], classes)
    labels_bayes_3a = bayes(m, sigma, p, data_3a["x"])
    score_bayes_3a = calculScore(labels_bayes_3a, data_3a["clasapp"][0])
    print("Naive Bayes : ", score_bayes_3a/len(labels_bayes_3a)*100)

    ## 2.4. Influence de la taille de l'ensemble d'apprentissage
    print("Partie 2.4.")
    data_3b = loadmat("p1_data3b.mat")

    # K plus proches voisins
    classes = [1] * 150
    classes += [2] * 150
    classes += [3] * 150
    for i in range(8):
        labels_kppv_3b = kppv(data_3b["test"], classes, 2*i+1, data_3b["x"])
        score_kppv_3b = calculScore(labels_kppv_3b, data_3b["clasapp"][0])
        print("k : ", 2*i+1, "score : ", score_kppv_3b/len(labels_kppv_3b)*100)

    # Naive Bayes
    m, sigma, p = entrainementBayes(data_3b["test"], classes)
    labels_bayes_3b = bayes(m, sigma, p, data_3b["x"])
    score_bayes_3b = calculScore(labels_bayes_3b, data_3b["clasapp"][0])
    print("Naive Bayes : ", score_bayes_3b/len(labels_bayes_3b)*100)

    ## 2.5. Distribution inconnue
    print("Partie 2.5.")
    data_4 = loadmat("p1_data4.mat")

    # K plus proches voisins
    classes = [1] * 70
    classes += [2] * 70
    classes += [3] * 70
    for i in range(8):
        labels_kppv_4 = kppv(data_4["test"], classes, 2*i+1, data_4["x"])
        score_kppv_4 = calculScore(labels_kppv_4, data_4["clasapp"][0])
        print("k : ", 2*i+1, "score : ", score_kppv_4/len(labels_kppv_4)*100)

    # Naive Bayes
    m, sigma, p = entrainementBayes(data_4["test"], classes)
    labels_bayes_4 = bayes(m, sigma, p, data_4["x"])
    score_bayes_4 = calculScore(labels_bayes_4, data_4["clasapp"][0])
    print("Naive Bayes : ", score_bayes_4/len(labels_bayes_4)*100)
