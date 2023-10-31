"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

from helpers.ClassificationData import ClassificationData
import helpers.analysis as an
import helpers.classifiers as classifiers

from keras.optimizers import Adam,SGD
import numpy as np
import keras as K


#######################################
def problematique_APP2():
    data3classes = ClassificationData()

    if True:
        print('\n\n=========================\nDonnées originales\n')
        data3classes.getStats(gen_print=True)
        data3classes.getBorders(view=True)

    if True:
        # Exemple de RN
        n_neurons = 15
        n_layers = 4

        nn1 = classifiers.NNClassify_APP2(data2train=data3classes, data2test=data3classes,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='relu',
                                          outputActivation='softmax', optimizer=SGD(learning_rate=0.001, momentum= 0.65),
                                          loss='categorical_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[
                                              K.callbacks.EarlyStopping(patience=100, verbose=1, restore_best_weights=1),
                                              classifiers.print_every_N_epochs(25)],
                                          # TODO à compléter L2.E4
                                          experiment_title='NN Simple',
                                          n_epochs=10000, savename='3classes',
                                          ndonnees_random=5000, gen_output=True, view=True)

    if True:  # TODO L3.E2
        ## Exemples de ppv avec ou sans k-moy
        ## 1-PPV avec comme représentants de classes l'ensemble des points déjà classés
        ppv5 = classifiers.PPVClassify_APP2(data2train=data3classes, data2test=data3classes, n_neighbors=1,
                                            experiment_title='1-PPV avec données orig comme représentants',
                                            gen_output=True, view=True)
        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes


        ppv1km1 = classifiers.PPVClassify_APP2(data2train=data3classes, data2test=data3classes,
                                               n_neighbors=1,
                                               experiment_title='1-PPV sur le 1-moy',
                                               useKmean=True, n_representants=35,
                                               gen_output=True, view=True)

    if True:  # TODO L3.E3
        # Exemple de classification bayésienne
        apriori = [1 / 3, 1 / 3, 1 / 3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
        bg1 = classifiers.BayesClassify_APP2(data2train=data3classes, data2test=data3classes,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités échantillonage',
                                             gen_output=True, view=True)
    plt.show()
    plt.waitforbuttonpress()


######################################
if __name__ == '__main__':
    problematique_APP2()
