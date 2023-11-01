"""
Authors: Antoine Marion 2021, JBM major refactor 2021, 2022, 2023

Fonctions de classification pour le labo
S8 GIA APP2
TODO modifier si nécessaire pour la problématique

Fichier comporte 3 grandes catégories de classes
A) Classificateurs de base: fonctions au prototype relativement similaire, prennent des données d'entraînement
        pour créer un modèle, et ensuite calculent la prédiction à partir de ce modèle
        suivent la métho générale d'apprentissage par modèle: construction, entraînement (fit), calculs (predict)

        *** Exceptionnellement, pour nous simplifier la vie ici, tous les predicts incluent le calcul des erreurs de
            prédiction, alors que fonctionnellement ça devrait plus aller dans le wrapper, voir B) ci-dessous. De cette
            manière, chaque classificateur instancié contient une trace de ses erreurs de prédiction, ce qui simplifie
            la traçabilité si on décide de sauver l'objet qui contient alors la performance en même temps que tous
            ses hyperparamètres.

    i. Bayes_Classifier: utilise une fonction de densité de probabilité (GaussianProbDensity ou histProbDensity)
        et le risque de Bayes pour classer. Implémenter mitaine pour tenir compte des coûts et des apriori.
    ii. PPV_Classifier: exécute un k-ppv, optionnellement effectue du clustering (Clusterer_APP2) sur les données
        fournies pour déterminer les représentants de classe. Implémenté avec sklearn.
    iii. NN_Classifier: modèle par réseau de neurones, nécessite un processus un peu plus complexe (preprocessing du data,
        définition d'architecture, entraînement). Option de sauvegarder les modèles et de les recharger lors de la
        prédiction. Implémenté avec keras.

    iv. KMeanAlgo: Pas un classificateur, mais même principe, effectue le clustering des k-moy pour 1 ou plusieurs jeux
        de points. Implémenté avec sklearn.

B) Wrappers pour les classificateurs de base pour l'APP (BayesClassify_APP2, PPVCLassify_APP2, NNClassify_APP2)
    fonctions aux prototypes relativement similaires, enveloppent les classificateurs ci-dessus avec du nice-to-have,
        en particulier d'affichage, visualisation de frontières. Réalisent un exemple complet de classification.
        prototype général: options d'algo, données d'entraînement + étiquettes, données de test aléatoires pour
            visualiser les frontières, options de graphique.
    Tous les wrappers requièrent des données sous forme d'objet helpers.ClassificationData, autant pour l'entraînement
        que les prédictions.
    Tous les wrappers affichent les frontières obtenues via la classification d'un nuage de points arbitraires calculé
        sur l'extent des données de classes.
    Certains wrappers acceptent un deuxième testset, utilisé ici pour reclasser les données d'origine, quand ça a du sens.

C) Autres fonctions modulaires
    i.  helpers pour modèles gaussiens
        - get_gaussian_borders: permet de calculer l'équation de la frontière entre chaque paire de classes d'entrée,
                en assumant un modèle gaussien; voir l'exercice préparatoire du laboratoire
    ii. helpers pour les RN
        - print_every_N_epochs: callback custom pour un affichage plus convivial pendant l'entraînement
"""

from itertools import combinations
import numpy as np
from enum import Flag, auto
import pickle
import os

from sklearn.cluster import KMeans as KM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import OneHotEncoder

import keras as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import helpers.analysis as an


###############################################
# Fonctions de densité de probabilité utilisées pour le classificateur par risque de Bayes
# Doivent avoir le prototype suivant:
#   - un constructeur qui accepte les données qui servent à construire le modèle de 1 seule classe, sous forme de Liste
#   - une méthode computeProbability, qui accepte une liste de points dont on veut calculer la probabilité
#     d'appartenance et qui retourne un vecteur de même longueur

class GaussianProbDensity:
    """
    Classe "virtuelle" appelée par BayesClassifier
    Modèle de classe gaussien
    Train intégré dans le constructeur
    Predict à part -> computeProbaility
    """
    def __init__(self, data2train):
        _, self.representationDimensions = np.asarray(data2train).shape
        self.mean, self.cov, _, _ = an.calcModeleGaussien(data2train)
        self.det = np.linalg.det(self.cov)
        assert self.det  # Prévient les erreurs si det = 0, normalement impossible mais bon
        self.inv_cov = np.linalg.inv(self.cov)

    def computeProbability(self, testdata1array):
        testDataNSamples, testDataDimensions = np.asarray(testdata1array).shape
        assert testDataDimensions == self.representationDimensions
        # calcule la distance de mahalanobis
        # itère sur l'ensemble des éléments à tester
        temp = np.array([testdata1array[j] - self.mean for j in range(testDataNSamples)])
        mahalanobis = np.array([np.matmul(np.matmul(temp[j], self.inv_cov), temp[j].T) for j in range(testDataNSamples)])
        return 1 / np.sqrt(self.det * (2 * np.pi) ** self.representationDimensions) * np.exp(-mahalanobis / 2)


class histProbDensity:
    """
    Classe "virtuelle" appelée par BayesClassifier
    Modèle de classe arbitraire (histogramme)
    Fonctionne juste pour une représentation 2D pour l'instant
    Train intégré dans le constructeur
    Predict à part -> computeProbaility
    """
    def __init__(self, data2train, title='', view=False):
        _, self.representationDimensions = np.asarray(data2train).shape
        # TODO problématique: modifier la modélisation pour fonctionner avec une dimensionalité plus élevée
        self.hist, self.xedges, self.yedges, self.zedges = an.creer_hist2D(data2train, title=title, view=view)

    def computeProbability(self, testdata1array):
        testDataNSamples, testDataDimensions = np.asarray(testdata1array).shape
        assert testDataDimensions == self.representationDimensions
        probabilities = np.zeros((testDataNSamples, 1))
        for row in range(0,len(testdata1array)):
            x_bin = np.searchsorted(self.xedges, testdata1array[row][0], side='right')-1
            y_bin = np.searchsorted(self.yedges, testdata1array[row][1], side='right')-1
            z_bin = np.searchsorted(self.zedges, testdata1array[row][2], side='right')-1

            if x_bin == -1 or x_bin == 59 or y_bin ==-1 or y_bin == 59 or z_bin ==-1 or z_bin == 59 :
                prob_point = 0
                probabilities[row] += float(prob_point)
            else:
                prob_point = self.hist[x_bin, y_bin, z_bin] / np.sum(self.hist)
                probabilities[row] += float(prob_point)

        return  probabilities


#############################################################################
# Good morning Bayes
class BayesClassifier:
    """
    Classificateur de Bayes
    Train() est intégré dans le constructeur, i.e. le constructeur calcule les modèles directement
    Predict est incomplet (ne tient pas compte du coût et des a priori
    """
    def __init__(self, data2trainLists, probabilitydensityType=GaussianProbDensity, apriori=None, costs=None):
        """
        data2trainLists: correspond au format de listes de ClassificationData()
        probailitydensityType: pointeur à une des fonctions de probabilité voir ci-dessus
        """
        self.densities = []  # Liste des densités de prob de chaque classe
        self.n_classes, _, self.representationDimensions = np.asarray(data2trainLists).shape
        if apriori:
            assert len(apriori) == self.n_classes
            self.apriori = np.array(apriori)
            assert np.sum(self.apriori) == 1
        else:
            self.apriori = np.ones((self.n_classes, 1)) / self.n_classes
        if costs:
            _x, _y = np.asarray(costs).shape
            assert _x == self.n_classes
            assert _y == self.n_classes
            self.costs = costs
        else:
            self.costs = np.ones((self.n_classes, self.n_classes)) - np.identity(self.n_classes)
        # Training happens here, calcul des modèles pour chaque classe
        for i in range(self.n_classes):
            self.densities.append(probabilitydensityType(data2trainLists[i]))

    def predict(self, testdata1array, expected_labels1array=None, gen_output=False):
        """
        testdata1array: correspond au format où toutes les données sont dans 1 seule liste peu importe la classe,
            voir ClassificationData()
        """
        testDataNSamples, testDataDimensions = np.asarray(testdata1array).shape
        assert testDataDimensions == self.representationDimensions
        classProbDensities = []
        # calcule la valeur de la probabilité d'appartenance à chaque classe pour les données à tester
        for i in range(self.n_classes):  # itère sur toutes les classes
            classProbDensities.append(self.densities[i].computeProbability(testdata1array) * float(self.apriori[i]))
        # reshape pour que les lignes soient les calculs pour 1 point original, i.e. même disposition que l'array d'entrée
        classProbDensities = np.array(classProbDensities).T
        classProbDensities = np.squeeze(classProbDensities)
        risques = []
        for row in classProbDensities:
            R1 = row[1] * self.costs[0][1] + row[2]* self.costs[0][2]
            R2 = row[0] * self.costs[1][0] + row[2]* self.costs[1][2]
            R3 = row[0] * self.costs[2][0] + row[1]* self.costs[2][1]
            risques.append([R1,R2,R3])
        # TODO problematique: take apriori and cost finto consideration! here for risk computation argmax assumes equal costs and apriori
        predictions = np.argmin(risques, axis=1).reshape(testDataNSamples, 1)
        if np.asarray(expected_labels1array).any():
            errors_indexes = an.calc_erreur_classification(expected_labels1array, predictions, gen_output)
        else:
            errors_indexes = np.asarray([])
        return predictions, errors_indexes


class BayesClassify_APP2:
    def __init__(self, data2train, data2test=None, ndonnees_random=5000,
                 probabilitydensityType=histProbDensity, apriori=None, costs=None,
                 experiment_title='Bayes Classifier', gen_output=False, view=False):
        """
        Wrapper avec tous les nice to have pour un classificateur bayésien
        """
        print('\n\n=========================\nNouveau classificateur: '+experiment_title)
        self.classifier = BayesClassifier(data2train.dataLists, probabilitydensityType, apriori=apriori, costs=costs)
        self.donneesTestRandom = an.genDonneesTest(ndonnees_random, data2train.extent)
        self.predictRandom, _ = self.classifier.predict(self.donneesTestRandom)  # classifie les données de test1
        if np.asarray(data2test).any():   # classifie les données de test2 si présentes
            self.predictTest, self.error_indexes = \
                self.classifier.predict(data2test.data1arrayTest, data2test.label1arrayTest, gen_output=gen_output)
        else:
            self.predictTest = []
            self.error_indexes = []
        if view:
            an.view_classification_results(original_data=data2train.data1array, test1data=self.donneesTestRandom,
                                       test2data=data2test.data1arrayTest, test2errors=self.error_indexes,
                                       colors_original=data2train.labels1array, colors_test1=self.predictRandom,
                                       colors_test2=self.predictTest / an.error_class / 0.75,
                                       experiment_title=f'Classification de Bayes, {experiment_title}',
                                       title_original='Données originales',
                                       title_test1=f'Données aléatoires classées',
                                       title_test2='Données d\'origine reclassées',
                                       extent=data2train.extent)


class PPVClassifier:
    def __init__(self, data2train, n_neighbors=1, metric='minkowski',
                 useKmean=False, n_represantants=1, experiment_title='PPV Classifier', view=False):
        # TODO L2.E3.1 Compléter la logique pour utiliser la librairie ici
        # le 1 est suspect et il manque des arguments
        self.n_classes, _, self.representationDimensions = np.asarray(data2train.dataLists).shape
        self.kNN = KNN(n_neighbors)  # minkowski correspond à distance euclidienne lorsque le paramètre p = 2
        # Exécute un clustering pour calculer les représentants de classe si demandés
        if useKmean:
            assert n_represantants >= n_neighbors
            self.km = Clusterer_APP2(data2train, n_representants=n_represantants,
                                     experiment_title='Représentants pour ' + experiment_title, view=view)
            reprData = self.km.clusterer.cluster_centers
            reprLabel = self.km.clusterer.cluster_labels
        else:  # sinon utilise les données fournies telles quelles comme représentants
            reprData = data2train.data1array
            reprLabel = data2train.labels1array
        # train est dans le constructeur ici aussi
        self.kNN.fit(reprData, reprLabel.ravel())  # initialise les représentants avec leur label de classe

    def predict(self, testdata1array, expected_labels1array=None, gen_output=False):
        _, testDataDimensions = np.asarray(testdata1array).shape
        assert testDataDimensions == self.representationDimensions
        predictions = self.kNN.predict(testdata1array)
        predictions = predictions.reshape(len(testdata1array), 1)
        if np.asarray(expected_labels1array).any():
            errors_indexes = an.calc_erreur_classification(expected_labels1array, predictions, gen_output)
        else:
            errors_indexes = np.asarray([])
        return predictions, errors_indexes


class PPVClassify_APP2:
    def __init__(self, data2train, data2test=None, n_neighbors=1, metric='minkowski', ndonnees_random=5000,
                 useKmean=False, n_representants=1,
                 experiment_title='PPV Classifier', gen_output=False, view=False):
        print('\n\n=========================\nNouveau classificateur: '+experiment_title)
        self.classifier = PPVClassifier(data2train, n_neighbors=n_neighbors, metric=metric,
                                        useKmean=useKmean, n_represantants=n_representants, experiment_title=experiment_title,
                                        view=True)
        self.donneesTestRandom = an.genDonneesTest(ndonnees_random, data2train.extent)
        self.predictRandom, _ = self.classifier.predict(self.donneesTestRandom)  # classifie les données de test
        if np.asarray(data2test).any():   # classifie les données de test2 si présentes
            self.predictTest, self.error_indexes = \
                self.classifier.predict(data2test.data1arrayTest, data2test.label1arrayTest, gen_output=True)
        else:
            self.predictTest = []
            self.error_indexes = []
        if view:
            an.view_classification_results(original_data=data2train.data1array, test1data=self.donneesTestRandom,
                                       test2data=data2test.data1arrayTest
                                            if np.asarray(data2test).any() else None,
                                       test2errors=self.error_indexes,
                                       colors_original=data2train.labels1array, colors_test1=self.predictRandom,
                                       colors_test2=self.predictTest / an.error_class / 0.75
                                            if np.asarray(data2test).any() else None,
                                       experiment_title=experiment_title,
                                       title_original='Représentants de classe',
                                       title_test1=f'Données aléatoires classées {n_neighbors}-PPV',
                                       title_test2=f'Prédiction de {n_neighbors}-PPV, données originales'
                                            if np.asarray(data2test).any() else None,
                                       extent=data2train.extent)


class KMeanAlgo:
    """
    Classe "virtuelle" appelée par Cluster_APP2
    Accepte un objet ClassificationData en entrée
    Produit une liste unique des représentants de classe et de leur étiquette
    """
    def __init__(self, data2train, n_representants=1):
        self.n_classes, _, _ = np.asarray(data2train.dataLists).shape
        self.kmeans_on_each_class = []
        self.cluster_centers = []
        self.cluster_labels = np.zeros((n_representants * self.n_classes, 1))
        for i in range(self.n_classes):  # itère sur l'ensemble des classes
            # TODO L2.E3.3 compléter la logique pour utiliser la librairie ici
            # encore une fois le 1 est suspect
            self.kmeans_on_each_class.append(KM(n_representants, n_init='auto'))
            self.kmeans_on_each_class[i].fit(np.array(data2train.dataLists[i]))
            self.cluster_centers.append(self.kmeans_on_each_class[i].cluster_centers_)
            self.cluster_labels[range(n_representants * i, n_representants * (i + 1))] = \
                data2train.labelsLists[i][0]  # assigne la classe en ordre ordinal croissant

        if n_representants == 1:  # gère les désagréments Python
            self.cluster_centers = np.array(self.cluster_centers)[:, 0]
        else:
            self.cluster_centers = np.array(self.cluster_centers)
            _x, _y, _z = self.cluster_centers.shape
            self.cluster_centers = self.cluster_centers.reshape(_x * _y, _z)


class Clusterer_APP2:
    def __init__(self, data2train, clusterer=KMeanAlgo, n_representants=1, experiment_title='Kmeans', view=False):
        self.n_classes, _, _ = np.asarray(data2train.dataLists).shape
        self.clusterer = clusterer(data2train, n_representants=n_representants)
        if view:
            an.view_classification_results(original_data=data2train.data1array, test1data=self.clusterer.cluster_centers,
                                           colors_original=data2train.labels1array, colors_test1=self.clusterer.cluster_labels,
                                           experiment_title=experiment_title,
                                           title_original='Données d\'origine',
                                           title_test1=f'Clustering de {n_representants}-Means',
                                           extent=data2train.extent)


class NNClassifier:
    class NNstate(Flag):
        """
        Énumération d'états pour suivre la consistance du classificateur au fur et à mesure des étapes
        Au contraire des autres algorithmes de l'APP, cette classe a un constructeur, preprocess, arch, fit et predict
            distincts.
        """
        constructed = auto()
        data_avail = auto()
        architecture = auto()
        trained = auto()

    def __init__(self):
        self.inputDimensions = 0
        self.outputDimensions = 0
        self.n_classes = 0
        self.minmax = (0, 0)
        self.traindata1array = np.asarray([])
        self.trainlabels1array = np.asarray([])
        self.validdata1array = np.asarray([])
        self.validlabels1array = np.asarray([])
        self.encoder = OneHotEncoder(sparse_output=False)
        self.NNmodel = Sequential()
        self.state = NNClassifier.NNstate.constructed
        return

    def preprocess_training_data(self, dataLists, labelsLists, train_fraction=0.8):
        # Housekeeping
        in_nclasses, in_nsamples, in_Dimensions = np.asarray(dataLists).shape
        out_nclasses, out_nsamples, out_Dimensions = np.asarray(labelsLists).shape
        assert in_nclasses == out_nclasses
        assert in_nsamples == out_nsamples
        if NNClassifier.NNstate.architecture in self.state:
            # If new data is not same dimension as before, invalidates previous arch & training
            if (in_Dimensions != self.inputDimensions) | (out_Dimensions != self.outputDimensions):
                print("Warning: new dataset has reset NN architecture")
                self.NNmodel = Sequential()
                self.state = NNClassifier.NNstate.constructed
        self.n_classes = in_nclasses
        self.inputDimensions = in_Dimensions
        self.outputDimensions = out_Dimensions

        # Preprocess (encode) labels
        temp_labels1array = np.asarray(labelsLists).reshape(self.n_classes * out_nsamples, self.outputDimensions)
        encodedLabels1array = self.encoder.fit_transform(temp_labels1array.reshape(-1, 1))
        encodedLabelsLists = []
        for i in range(self.n_classes):
            encodedLabelsLists.append(
                encodedLabels1array[range(i * len(labelsLists[i]), (i + 1) * len(labelsLists[i]))])

        # Preprocess Data
        # TODO problématique: voir si d'autres transformations sont requises, ajuster predict en conséquence
        scaledData, self.minmax = an.scaleData(dataLists)

        # Split into train and validation subsets
        self.traindata1array, self.trainlabels1array, self.validdata1array, self.validlabels1array = \
            an.splitDataNN(self.n_classes, scaledData, encodedLabelsLists, train_fraction=train_fraction)

        # Housekeeping
        if self.state == NNClassifier.NNstate.constructed:
            self.state = NNClassifier.NNstate.data_avail
        else:
            self.state = self.state | NNClassifier.NNstate.data_avail

    def init_model(self, n_neurons, n_hidden_layers, innerActivation='tanh', outputActivation='relu',
                   optimizer=Adam(), loss='binary_crossentropy', metrics=None, gen_output=False):
        assert NNClassifier.NNstate.data_avail in self.state
        if (NNClassifier.NNstate.trained in self.state) | (NNClassifier.NNstate.architecture in self.state):
            print("Warning: architecture redefinition resets previous one")
            self.NNmodel = Sequential()
            self.state = self.state and not NNClassifier.NNstate.trained
            self.state = self.state and not NNClassifier.NNstate.architecture
        self.NNmodel.add(Dense(units=n_neurons, activation=innerActivation, input_shape=(self.traindata1array.shape[-1],)))
        for i in range(2, n_hidden_layers):
            self.NNmodel.add(Dense(units=n_neurons, activation=innerActivation))
        self.NNmodel.add(Dense(units=self.trainlabels1array.shape[-1], activation=outputActivation))
        self.NNmodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if gen_output:
            print(self.NNmodel.summary())
        self.state = self.state | NNClassifier.NNstate.architecture
        return

    def train_model(self, n_epochs, batch_size=None, callback_list=None, savename='', view=False):
        assert NNClassifier.NNstate.data_avail in self.state
        assert NNClassifier.NNstate.architecture in self.state
        if batch_size is None:
            batch_size = len(self.traindata1array)
        self.NNmodel.fit(self.traindata1array, self.trainlabels1array, batch_size=batch_size, verbose=1,
                    epochs=n_epochs, shuffle=True, callbacks=callback_list,
                    validation_data=(self.validdata1array, self.validlabels1array))

        # Save trained model to disk
        if savename:
            self.NNmodel.save('saves'+os.sep+savename+'.keras')
            pickle.dump([self.minmax, self.NNmodel.history], open('saves'+os.sep+savename+'.pkl','wb'))
        if view:
            an.plot_metrics(self.NNmodel)

        self.state = self.state | NNClassifier.NNstate.trained

    def predict(self, testdata1array, expected_labels1array=None, savename='', gen_output=False):
        # Ce mécanisme permet de recharger un modèle déjà entraîné du disque et d'utiliser predict direct
        # sans passer par le reste de la logique d'initialisation
        if savename:
            self.NNmodel = self.NNmodel.load('saves'+os.sep+savename+'.keras')
            self.inputDimensions = self.NNmodel.input_shape[1]
            self.outputDimensions = self.NNmodel.output_shape[1]
            self.minmax, self.NNmodel.history = pickle.load(open('saves'+os.sep+savename+'.pkl', 'rb'))
            self.state = NNClassifier.NNstate.architecture
            if self.NNmodel.history:
                self.state = self.state | NNClassifier.NNstate.trained
        assert NNClassifier.NNstate.trained in self.state

        testnsamples, testinputDimensions = np.asarray(testdata1array).shape
        assert testinputDimensions == self.inputDimensions
        if np.asarray(expected_labels1array).any():
            if savename:
                tempencoded = self.encoder.fit_transform(np.array(expected_labels1array).reshape(-1, 1))
                _, testoutputDimensions = np.asarray(tempencoded).shape
                assert testoutputDimensions == self.outputDimensions  # ensure minimal compatibility, not a very strict test
                _, self.outputDimensions = np.asarray(expected_labels1array).shape
            else:
                _, testoutputDimensions = np.asarray(expected_labels1array).shape
                assert testoutputDimensions == self.outputDimensions

        # classifie les données de test
        # il faut refaire le même preprocess sur les données de test que d'entraînement
        # decode la sortie one hot en numéro de classe 0 à N directement
        predictions = np.argmax(self.NNmodel.predict(an.scaleDataKnownMinMax(testdata1array, self.minmax)), axis=1)
        predictions = predictions.reshape(testnsamples, 1)
        if np.asarray(expected_labels1array).any():
            errors_indexes = an.calc_erreur_classification(expected_labels1array, predictions, gen_output)
        else:
            errors_indexes = np.array([])
        return predictions, errors_indexes


class NNClassify_APP2:
    def __init__(self, data2train, data2test, n_layers, n_neurons, innerActivation='tanh', outputActivation='softmax',
                 optimizer=Adam(), loss='binary_crossentropy', metrics=None,
                 callback_list=None, n_epochs=1000, savename='', ndonnees_random=5000,
                 experiment_title='NN Classifier', gen_output=True, view=True):
        print('\n\n=========================\nNouveau classificateur: ' + experiment_title)
        self.classifier = NNClassifier()
        self.classifier.preprocess_training_data(dataLists=data2train.dataLists, labelsLists=data2train.labelsLists)
        self.classifier.init_model(n_neurons, n_layers, innerActivation=innerActivation,
                                   outputActivation=outputActivation, gen_output=gen_output,
                                   optimizer=optimizer, loss=loss, metrics=metrics)
        self.classifier.train_model(n_epochs, callback_list=callback_list, batch_size= 50, savename=savename, view=view)
        plt.show()
        self.donneesTestRandom = an.genDonneesTest(ndonnees_random, data2train.extent)
        self.predictRandom, _ = self.classifier.predict(testdata1array=self.donneesTestRandom)
        self.predictTest, self.error_indexes = self.classifier.predict(testdata1array=data2test.data1arrayTest,
                                                                       expected_labels1array=data2test.label1arrayTest,
                                                                       gen_output=gen_output)
        if view:
            an.view_classification_results(original_data=data2train.data1array, test1data=self.donneesTestRandom,
                                           test2data=data2test.data1arrayTest,
                                           test2errors=self.error_indexes,
                                           colors_original=data2train.labels1array, colors_test1=self.predictRandom,
                                           colors_test2=self.predictTest / an.error_class / 0.75,
                                           experiment_title=experiment_title + f'NN {n_layers} layer(s) caché(s), {n_neurons} neurones par couche',
                                           title_original='Données originales',
                                           title_test1=f'Données aléatoires classées par le RNA',
                                           title_test2='Prédiction du RNA, données originales',
                                           extent=data2train.extent)


def get_gaussian_borders(dataLists):
    """
    ***Pas validé sur des classes autres que les classes du laboratoire
    Calcule les frontières numériques entre n classes de dimension 2 en assumant un modèle gaussien

    data format: [C1, C2, C3, ... Cn]
    retourne 1 liste:
        border_coeffs: coefficients numériques des termes de l'équation de frontières
            [x**2, xy, y**2, x, y, cst (cote droit de l'equation de risque), cst (dans les distances de mahalanobis)]

    Le calcul repose sur une préparation analytique qui conduit à
    g(y) = y*A*y + b*y + C          avec
    y la matrice des dimensions d'1 vecteur de la classe
    et pour chaque paire de classe C1 C2:
    A = inv(cov_1) - inv(cov_2)
    b = -2*(inv(cov_2)*m2 - inv(cov_1)*m1)
    C = c+d
    d = -(transp(m1)*inv(cov_1)*m1 - transp(m2)*inv(cov_2)*m2)
    c = -ln(det(cov_2)/det(cov_1))
    """
    # Initialisation des listes
    # Portion numérique
    avg_list = []
    cov_list = []
    det_list = []
    inv_cov_list = []
    border_coeffs = []

    n_classes = len(dataLists)
    # calcul des stats des classes
    for i in range(n_classes):
        # stats de base
        avg, cov, _, _ = an.calcModeleGaussien(dataLists[i])
        avg_list.append(avg)
        cov_list.append(cov)
        inv_cov_list.append(np.linalg.inv(cov))
        det_list.append(np.linalg.det(cov))

    # calcul des frontières
    for classePair in combinations(range(n_classes), 2):
        # les coefficients sont tirés de la partie préparatoire du labo
        # i.e. de la résolution analytique du risque de Bayes
        # partie numérique
        a = np.array(inv_cov_list[classePair[1]] - inv_cov_list[classePair[0]])
        b = -np.array([2 * (np.dot(inv_cov_list[classePair[1]], avg_list[classePair[1]]) -
                            np.dot(inv_cov_list[classePair[0]], avg_list[classePair[0]]))])
        d = -(np.dot(np.dot(avg_list[classePair[0]], inv_cov_list[classePair[0]]), np.transpose(avg_list[classePair[0]])) -
              np.dot(np.dot(avg_list[classePair[1]], inv_cov_list[classePair[1]]), np.transpose(avg_list[classePair[1]])))
        c = -np.log(det_list[classePair[1]] / det_list[classePair[0]])

        # rappel: coef order: [x**2, xy, y**2, x, y, cst (cote droit log de l'equation de risque), cst (dans les distances de mahalanobis)]
        border_coeffs.append([a[0, 0], a[0, 1] + a[1, 0], a[1, 1], b[0, 0], b[0, 1], c, d])
        # print(border_coeffs[-1])

    return border_coeffs


class print_every_N_epochs(K.callbacks.Callback):
    """
    Helper callback pour remplacer l'affichage lors de l'entraînement
    """
    def __init__(self, N_epochs):
        super().__init__()
        self.epochs = N_epochs

    def on_epoch_end(self, epoch, logs=None):
        # TODO L2.E2.4
        if (int(epoch)) == 0:
            print("Epoch: {:>3} | Loss: ".format(epoch) +
                  f"{logs['loss']:.4e}" + " | Valid loss: " + f"{logs['val_loss']:.4e}" +
                  (f" | Accuracy: {logs['accuracy']:.4e}" + " | Valid accuracy " + f"{logs['val_accuracy']:.4e}"
                   if 'accuracy' in logs else "") )
