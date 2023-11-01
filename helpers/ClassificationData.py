"""
Classe "ClassificationData" : vise à contenir la représentation à utiliser pour classifier par la suite
Constructeur:
    par défaut, charge les 3 classes du laboratoire
    peut aussi utiliser un array ou une liste de listes
Membres :
    pour simplifier l'utilisation dans différents modes d'entraînement des techniques et dans les prédictions subséquentes,
    les données sont représentées sous 2 formes
        i. Liste de listes ou array_like
            -> Format nécessaire quand on veut constuire l'objet à partir de données existantes.
                TODO problématique: construire un object classification data à partir des quantités retenues pour la représentation
            -> dataLists : array_like de dimension K * L * M, K le nombre de classes distinctes, L les données pour chaque classe
            Note: il n'existe pas d'étiquette explicite pour ce format, on utilise automatiquement la position dans l'array selon la première dimension
        ii. 1 seul long vecteur. Généré automatiquement à partir du format précédent
            -> data1array: array N x M où N est le nombre total de données disponibles (N = K * L)
                et M la dimension de l'espace de représentation
            -> labels1array: étiquettes de classes. Entiers de 0 à L-1 générés automatiquement à partir de la dimension L dans dataLists
    extent: la plage utile des données
Méthodes:
    getStats: calcule des statistiques descriptives sur chaque classe
    getBorders: visualise les nuages de points de chaque classe avec ou sans les frontières calculées
"""

import numpy as np
import os
import helpers.analysis as an
import helpers.classifiers as classifiers
from sklearn.model_selection import train_test_split as ttsplit

class ClassificationData:

    def __init__(self, existingData=None, Test = False):
        if np.asarray(existingData).any():
            self.dataLists = existingData  # TODO JB assert  le format fourni
        else:
            self.dataLists = []
            # Import data from text files in subdir
            self.dataLists.append(np.loadtxt('data'+os.sep+'data_3classes_app'+os.sep+'C1.txt'))
            self.dataLists.append(np.loadtxt('data'+os.sep+'data_3classes_app'+os.sep+'C2.txt'))
            self.dataLists.append(np.loadtxt('data'+os.sep+'data_3classes_app'+os.sep+'C3.txt'))

        histo = classifiers.histProbDensity(self.dataLists[1])
        x_size, y_size, z_size = histo.hist.shape

        nombre_de_points = len(self.dataLists[0]) - len(self.dataLists[1])
        points_aleatoires = []
        while len(points_aleatoires) < nombre_de_points:

            index = np.unravel_index(np.random.choice(x_size * y_size * z_size, p=histo.hist.flatten()),
                                     (x_size, y_size, z_size))

            x = np.random.uniform(index[0] / x_size, (index[0] + 1) / x_size)  # Normalisation entre 0 et 1
            y = np.random.uniform(index[1] / y_size, (index[1] + 1) / y_size)  # Normalisation entre 0 et 1
            z = np.random.uniform(index[2] / z_size, (index[2] + 1) / z_size)

            if np.random.uniform(0, 1) < histo.hist[tuple(index)]:
                points_aleatoires.append([x, y, z])
        points_generated_array = np.array(points_aleatoires)

        self.dataLists[1] = np.concatenate((self.dataLists[1], points_generated_array))

        histo = classifiers.histProbDensity(self.dataLists[2])
        x_size, y_size, z_size = histo.hist.shape

        nombre_de_points = len(self.dataLists[0]) - len(self.dataLists[2])
        points_aleatoires = []
        while len(points_aleatoires) < nombre_de_points:

            index = np.unravel_index(np.random.choice(x_size * y_size * z_size, p=histo.hist.flatten()),
                                     (x_size, y_size, z_size))

            x = np.random.uniform(index[0] / x_size, (index[0] + 1) / x_size)  # Normalisation entre 0 et 1
            y = np.random.uniform(index[1] / y_size, (index[1] + 1) / y_size)  # Normalisation entre 0 et 1
            z = np.random.uniform(index[2] / z_size, (index[2] + 1) / z_size)

            if np.random.uniform(0, 1) < histo.hist[tuple(index)]:
                points_aleatoires.append([x, y, z])
        points_generated_array = np.array(points_aleatoires)

        self.dataLists[2] = np.concatenate((self.dataLists[2], points_generated_array))

        #instanciate the shape of the data lists for validation and tests
        self.dataListValidation = self.dataLists
        self.dataListsTest = self.dataLists

        #defines the labels to split for the first class
        C1 = [0]*len(self.dataLists[0])
        C2 = [1]*len(self.dataLists[1])
        C3 = [2]*len(self.dataLists[2])

        X_train, self.dataListValidation[0], y_train, y_test = ttsplit(self.dataLists[0], C1, test_size=0.05)
        self.dataLists[0], self.dataListsTest[0], y_train, y_val = ttsplit(X_train, y_train, test_size=0.2)

        X_train, self.dataListValidation[1], y_train, y_test = ttsplit(self.dataLists[1], C2, test_size=0.05)
        self.dataLists[1], self.dataListsTest[1], y_train, y_val = ttsplit(X_train, y_train, test_size=0.2)

        X_train, self.dataListValidation[2], y_train, y_test = ttsplit(self.dataLists[2], C3, test_size=0.05)
        self.dataLists[2], self.dataListsTest[2], y_train, y_val = ttsplit(X_train, y_train, test_size=0.2)

        # Training
        # reorganisation en 1 seul vecteur pour certains entraînements et les predicts
        self.dataLists = np.array(self.dataLists)
        self._x, self._y, self._z = self.dataLists.shape
        # Chaque ligne de data contient 1 point en 2D
        # Les points des 3 classes sont mis à la suite en 1 seul long array
        self.data1array = self.dataLists.reshape(self._x * self._y, self._z)
        self.ndata = len(self.data1array)

        # assignation des classes d'origine 0 à 2 pour C1 à C3 respectivement
        self.labels1array = np.zeros([self.ndata, 1])
        self.labelsLists = []
        self.labelsLists.append(self.labels1array[range(len(self.dataLists[0]))])
        for i in range(1,self._x):
            self.labels1array[range(i * len(self.dataLists[i]), (i + 1) * len(self.dataLists[i]))] = i
            self.labelsLists.append(self.labels1array[range(i * len(self.dataLists[i]), (i + 1) * len(self.dataLists[i]))])


        # Validation
        # reorganisation en 1 seul vecteur pour certains entraînements et les predicts
        self.dataListValidation = np.array(self.dataListValidation)
        self._xVal, self._yVal, self._zVal = self.dataListValidation.shape
        # Chaque ligne de data contient 1 point en 2D
        # Les points des 3 classes sont mis à la suite en 1 seul long array
        self.data1arrayValidation = self.dataListValidation.reshape(self._xVal * self._yVal, self._zVal)
        self.ndataValidation = len(self.data1arrayValidation)

        # assignation des classes d'origine 0 à 2 pour C1 à C3 respectivement
        self.labels1arrayValidation = np.zeros([self.ndataValidation, 1])
        self.labelsListsValidation = []
        self.labelsListsValidation.append(self.labels1arrayValidation[range(len(self.dataListValidation[0]))])
        for i in range(1,self._x):
            self.labels1arrayValidation[range(i * len(self.dataListValidation[i]), (i + 1) * len(self.dataListValidation[i]))] = i
            self.labelsListsValidation.append(self.labels1arrayValidation[range(i * len(self.dataListValidation[i]), (i + 1) * len(self.dataListValidation[i]))])


        # Test
        # reorganisation en 1 seul vecteur pour certains entraînements et les predicts
        self.dataListsTest = np.array(self.dataListsTest)
        self._xTest, self._yTest, self._zTest = self.dataListsTest.shape
        # Chaque ligne de data contient 1 point en 2D
        # Les points des 3 classes sont mis à la suite en 1 seul long array
        self.data1arrayTest = self.dataListsTest.reshape(self._xTest * self._yTest, self._zTest)
        self.ndataTest = len(self.data1arrayTest)

        # assignation des classes d'origine 0 à 2 pour C1 à C3 respectivement
        self.label1arrayTest = np.zeros([self.ndataTest, 1])
        self.labelsListsTest = []
        self.labelsListsTest.append(self.label1arrayTest[range(len(self.dataListsTest[0]))])
        for i in range(1,self._x):
            self.label1arrayTest[range(i * len(self.dataListsTest[i]), (i + 1) * len(self.dataListsTest[i]))] = i
            self.labelsListsTest.append(self.label1arrayTest[range(i * len(self.dataListsTest[i]), (i + 1) * len(self.dataListsTest[i]))])

        # Min et max des données
        self.extent = an.Extent(ptList=self.data1array)
        self.extentVal = an.Extent(ptList=self.data1arrayValidation)
        self.extentTest = an.Extent(ptList=self.data1arrayTest)

        self.m = []
        self.cov = []
        self.valpr = []
        self.vectpr = []
        self.coeffs = []

        self.getStats()
        self.getBorders()

    def getStats(self, gen_print=False):
        if not self.m:
            for i in range(self._x):
                _m, _cov, _valpr, _vectpr = an.calcModeleGaussien(self.dataLists[i])
                self.m.append(_m)
                self.cov.append(_cov)
                self.valpr.append(_valpr)
                self.vectpr.append(_vectpr)
        if gen_print:
            for i in range(self._x):
                an.printModeleGaussien(
                    self.m[i], self.cov[i], self.valpr[i], self.vectpr[i], f'\nClasse {i + 1}' if gen_print else '')
        return self.m, self.cov, self.valpr, self.vectpr

    def getBorders(self, view=False):
        if not self.coeffs:
            self.coeffs = classifiers.get_gaussian_borders(self.dataLists)
        if view:
            an.view_classes(self.dataLists, self.extent, self.coeffs)
        return self.coeffs