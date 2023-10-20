"""
Classe "ImageCollection" pour charger et visualiser les images de la problématique
Membres :
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, changer le flag load_all du constructeur à True)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes pour la problématique :
    generateRGBHistograms : calcul l'histogramme RGB de chaque image, à compléter
    generateRepresentation : vide, à compléter pour la problématique
Méthodes génériques : TODO JB move to helpers
    generateHistogram : histogramme une image à 3 canaux de couleurs arbitraires
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from enum import IntEnum, auto

from skimage import color as skic
from skimage import io as skiio

import helpers.analysis as an


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """
    class imageLabels(IntEnum):
        coast = auto()
        forest = auto()
        street = auto()

    def __init__(self, load_all=False):
        # liste de toutes les images
        self.image_folder = r"data" + os.sep + "baseDeDonneesImages"
        self._path = glob.glob(self.image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in image_list if '.jpg' in i]

        self.all_images_loaded = False
        self.images = []

        # Crée un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        #            [Nombre image, hauteur, largeur, RGB]
        if load_all:
            self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            self.all_images_loaded = True

        self.labels = []
        for i in image_list:
            if 'coast' in i:
                self.labels.append(ImageCollection.imageLabels.coast)
            elif 'forest' in i:
                self.labels.append(ImageCollection.imageLabels.forest)
            elif 'street' in i:
                self.labels.append(ImageCollection.imageLabels.street)
            else:
                raise ValueError(i)

    def get_samples(self, N):
        return np.sort(random.sample(range(np.size(self.image_list, 0)), N))

    def generateHistogram(self, image, n_bins=256):
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = 3
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            for j in range(n_channels):
                pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)
        return pixel_values

    def generateRGBHistograms(self):
        """
        Calcule les histogrammes RGB de toutes les images
        """
        # TODO L1.E4.6 S'inspirer de view_histogrammes et déménager le code pertinent ici
        avg_mean_rgb = np.array(np.zeros((len(self.image_list),3)))
        avg_var_rgb = np.array(np.zeros((len(self.image_list),3)))
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            avg_mean_rgb[image_counter] = (np.mean(imageRGB, axis = (0,1)))
            avg_var_rgb[image_counter] = (np.var(imageRGB, axis = (0,1)))

        matrices_correlation = np.corrcoef(avg_mean_rgb.T, avg_var_rgb.T)
        print(matrices_correlation)
        print(avg_mean_rgb)

    def histAvgRegionAnalysisHVS(self):
        MeanhistvaluesHSVRegionLowIntensityCoast = []
        MeanhistvaluesHSVRegionMidIntensityCoast = []
        MeanhistvaluesHSVRegionHighIntensityCoast = []
        MeanhistvaluesHSVRegionTotalIntensityCoast = []
        MeanhistvaluesHSVRegionLowIntensityForest = []
        MeanhistvaluesHSVRegionMidIntensityForest = []
        MeanhistvaluesHSVRegionHighIntensityForest = []
        MeanhistvaluesHSVRegionTotalIntensityForest = []
        MeanhistvaluesHSVRegionLowIntensityStreet = []
        MeanhistvaluesHSVRegionMidIntensityStreet = []
        MeanhistvaluesHSVRegionHighIntensityStreet = []
        MeanhistvaluesHSVRegionTotalIntensityStreet = []
        VarhistvaluesHSVRegionLowIntensityCoast = []
        VarhistvaluesHSVRegionMidIntensityCoast = []
        VarhistvaluesHSVRegionHighIntensityCoast = []
        VarhistvaluesHSVRegionTotalIntensityCoast = []
        VarhistvaluesHSVRegionLowIntensityForest = []
        VarhistvaluesHSVRegionMidIntensityForest = []
        VarhistvaluesHSVRegionHighIntensityForest = []
        VarhistvaluesHSVRegionTotalIntensityForest = []
        VarhistvaluesHSVRegionLowIntensityStreet = []
        VarhistvaluesHSVRegionMidIntensityStreet = []
        VarhistvaluesHSVRegionHighIntensityStreet = []
        VarhistvaluesHSVRegionTotalIntensityStreet = []

        i = 0
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            #imageLabhist = an.rescaleHistLab(imageLab, n_bins)  # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            #histvaluesLAB = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)
            mean_r_low = (np.mean(histvaluesHSV[0][0:85]))
            mean_g_low = (np.mean(histvaluesHSV[1][0:85]))
            mean_b_low = (np.mean(histvaluesHSV[2][0:85]))
            mean_r_mid = (np.mean(histvaluesHSV[0][85:170]))
            mean_g_mid = (np.mean(histvaluesHSV[1][85:170]))
            mean_b_mid = (np.mean(histvaluesHSV[2][85:170]))
            mean_r_high = (np.mean(histvaluesHSV[0][170:255]))
            mean_g_high = (np.mean(histvaluesHSV[1][170:255]))
            mean_b_high = (np.mean(histvaluesHSV[2][170:255]))
            std_r_low = (np.var(histvaluesHSV[0][0:85]))
            std_g_low = (np.var(histvaluesHSV[1][0:85]))
            std_b_low = (np.var(histvaluesHSV[2][0:85]))
            std_r_mid = (np.var(histvaluesHSV[0][85:170]))
            std_g_mid = (np.var(histvaluesHSV[1][85:170]))
            std_b_mid = (np.var(histvaluesHSV[2][85:170]))
            std_r_high = (np.var(histvaluesHSV[0][170:255]))
            std_g_high = (np.var(histvaluesHSV[1][170:255]))
            std_b_high = (np.var(histvaluesHSV[2][170:255]))

            if "coast" in self.image_list[image_counter]:
                MeanhistvaluesHSVRegionLowIntensityCoast.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesHSVRegionMidIntensityCoast.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesHSVRegionHighIntensityCoast.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesHSVRegionTotalIntensityCoast.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesHSVRegionLowIntensityCoast.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesHSVRegionMidIntensityCoast.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesHSVRegionHighIntensityCoast.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesHSVRegionTotalIntensityCoast.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            if "forest" in self.image_list[image_counter]:
                MeanhistvaluesHSVRegionLowIntensityForest.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesHSVRegionMidIntensityForest.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesHSVRegionHighIntensityForest.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesHSVRegionTotalIntensityForest.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesHSVRegionLowIntensityForest.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesHSVRegionMidIntensityForest.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesHSVRegionHighIntensityForest.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesHSVRegionTotalIntensityForest.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            if "street" in self.image_list[image_counter]:
                MeanhistvaluesHSVRegionLowIntensityStreet.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesHSVRegionMidIntensityStreet.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesHSVRegionHighIntensityStreet.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesHSVRegionTotalIntensityStreet.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesHSVRegionLowIntensityStreet.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesHSVRegionMidIntensityStreet.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesHSVRegionHighIntensityStreet.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesHSVRegionTotalIntensityStreet.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            i = i + 1
            print(i)

        avgMeanhistvaluesHSVRegionLowIntensityCoast = np.mean(MeanhistvaluesHSVRegionLowIntensityCoast, axis=0)
        avgMeanhistvaluesHSVRegionMidIntensityCoast = np.mean(MeanhistvaluesHSVRegionMidIntensityCoast, axis=0)
        avgMeanhistvaluesHSVRegionHighIntensityCoast = np.mean(MeanhistvaluesHSVRegionHighIntensityCoast, axis=0)
        avgMeanhistvaluesHSVRegionTotalIntensityCoast = np.mean(MeanhistvaluesHSVRegionTotalIntensityCoast, axis=0)
        avgVarhistvaluesHSVRegionLowIntensityCoast = np.mean(VarhistvaluesHSVRegionLowIntensityCoast, axis=0)
        avgVarhistvaluesHSVRegionMidIntensityCoast = np.mean(VarhistvaluesHSVRegionMidIntensityCoast, axis=0)
        avgVarhistvaluesHSVRegionHighIntensityCoast = np.mean(VarhistvaluesHSVRegionHighIntensityCoast, axis=0)
        avgVarhistvaluesHSVRegionTotalIntensityCoast = np.mean(VarhistvaluesHSVRegionTotalIntensityCoast, axis=0)

        avgMeanhistvaluesHSVRegionLowIntensityForest = np.mean(MeanhistvaluesHSVRegionLowIntensityForest, axis=0)
        avgMeanhistvaluesHSVRegionMidIntensityForest = np.mean(MeanhistvaluesHSVRegionMidIntensityForest, axis=0)
        avgMeanhistvaluesHSVRegionHighIntensityForest = np.mean(MeanhistvaluesHSVRegionHighIntensityForest, axis=0)
        avgMeanhistvaluesHSVRegionTotalIntensityForest = np.mean(MeanhistvaluesHSVRegionTotalIntensityForest, axis=0)
        avgVarhistvaluesHSVRegionLowIntensityForest = np.mean(VarhistvaluesHSVRegionLowIntensityForest, axis=0)
        avgVarhistvaluesHSVRegionMidIntensityForest = np.mean(VarhistvaluesHSVRegionMidIntensityForest, axis=0)
        avgVarhistvaluesHSVRegionHighIntensityForest = np.mean(VarhistvaluesHSVRegionHighIntensityForest, axis=0)
        avgVarhistvaluesHSVRegionTotalIntensityForest = np.mean(VarhistvaluesHSVRegionTotalIntensityForest, axis=0)

        avgMeanhistvaluesHSVRegionLowIntensityStreet = np.mean(MeanhistvaluesHSVRegionLowIntensityStreet, axis=0)
        avgMeanhistvaluesHSVRegionMidIntensityStreet = np.mean(MeanhistvaluesHSVRegionMidIntensityStreet, axis=0)
        avgMeanhistvaluesHSVRegionHighIntensityStreet = np.mean(MeanhistvaluesHSVRegionHighIntensityStreet, axis=0)
        avgMeanhistvaluesHSVRegionTotalIntensityStreet = np.mean(MeanhistvaluesHSVRegionTotalIntensityStreet, axis=0)
        avgVarhistvaluesHSVRegionLowIntensityStreet = np.mean(VarhistvaluesHSVRegionLowIntensityStreet, axis=0)
        avgVarhistvaluesHSVRegionMidIntensityStreet = np.mean(VarhistvaluesHSVRegionMidIntensityStreet, axis=0)
        avgVarhistvaluesHSVRegionHighIntensityStreet = np.mean(VarhistvaluesHSVRegionHighIntensityStreet, axis=0)
        avgVarhistvaluesHSVRegionTotalIntensityStreet = np.mean(VarhistvaluesHSVRegionTotalIntensityStreet, axis=0)

        print("avgMeanhistvaluesHSVRegionLowIntensityCoast")
        print(avgMeanhistvaluesHSVRegionLowIntensityCoast)
        print("avgMeanhistvaluesHSVRegionMidIntensityCoast")
        print(avgMeanhistvaluesHSVRegionMidIntensityCoast)
        print("avgMeanhistvaluesHSVRegionHighIntensityCoast")
        print(avgMeanhistvaluesHSVRegionHighIntensityCoast)
        print("avgMeanhistvaluesHSVRegionTotalIntensityCoast")
        print(avgMeanhistvaluesHSVRegionTotalIntensityCoast)
        print("avgVarhistvaluesHSVRegionLowIntensityCoast")
        print(avgVarhistvaluesHSVRegionLowIntensityCoast)
        print("avgVarhistvaluesHSVRegionMidIntensityCoast")
        print(avgVarhistvaluesHSVRegionMidIntensityCoast)
        print("avgVarhistvaluesHSVRegionHighIntensityCoast")
        print(avgVarhistvaluesHSVRegionHighIntensityCoast)
        print("avgVarhistvaluesHSVRegionTotalIntensityCoast")
        print(avgVarhistvaluesHSVRegionTotalIntensityCoast)

        print("avgMeanhistvaluesHSVRegionLowIntensityForest")
        print(avgMeanhistvaluesHSVRegionLowIntensityForest)
        print("avgMeanhistvaluesHSVRegionMidIntensityForest")
        print(avgMeanhistvaluesHSVRegionMidIntensityForest)
        print("avgMeanhistvaluesHSVRegionHighIntensityForest")
        print(avgMeanhistvaluesHSVRegionHighIntensityForest)
        print("avgMeanhistvaluesHSVRegionTotalIntensityForest")
        print(avgMeanhistvaluesHSVRegionTotalIntensityForest)
        print("avgVarhistvaluesHSVRegionLowIntensityForest")
        print(avgVarhistvaluesHSVRegionLowIntensityForest)
        print("avgVarhistvaluesHSVRegionMidIntensityForest")
        print(avgVarhistvaluesHSVRegionMidIntensityForest)
        print("avgVarhistvaluesHSVRegionHighIntensityForest")
        print(avgVarhistvaluesHSVRegionHighIntensityForest)
        print("avgVarhistvaluesHSVRegionTotalIntensityForest")
        print(avgVarhistvaluesHSVRegionTotalIntensityForest)

        print("avgMeanhistvaluesHSVRegionLowIntensityStreet")
        print(avgMeanhistvaluesHSVRegionLowIntensityStreet)
        print("avgMeanhistvaluesHSVRegionMidIntensityStreet")
        print(avgMeanhistvaluesHSVRegionMidIntensityStreet)
        print("avgMeanhistvaluesHSVRegionHighIntensityStreet")
        print(avgMeanhistvaluesHSVRegionHighIntensityStreet)
        print("avgMeanhistvaluesHSVRegionTotalIntensityStreet")
        print(avgMeanhistvaluesHSVRegionTotalIntensityStreet)
        print("avgVarhistvaluesHSVRegionLowIntensityStreet")
        print(avgVarhistvaluesHSVRegionLowIntensityStreet)
        print("avgVarhistvaluesHSVRegionMidIntensityStreet")
        print(avgVarhistvaluesHSVRegionMidIntensityStreet)
        print("avgVarhistvaluesHSVRegionHighIntensityStreet")
        print(avgVarhistvaluesHSVRegionHighIntensityStreet)
        print("avgVarhistvaluesHSVRegionTotalIntensityStreet")
        print(avgVarhistvaluesHSVRegionTotalIntensityStreet)

    def histAvgRegionAnalysisLAB(self):
        MeanhistvaluesLABRegionLowIntensityCoast = []
        MeanhistvaluesLABRegionMidIntensityCoast = []
        MeanhistvaluesLABRegionHighIntensityCoast = []
        MeanhistvaluesLABRegionTotalIntensityCoast = []
        MeanhistvaluesLABRegionLowIntensityForest = []
        MeanhistvaluesLABRegionMidIntensityForest = []
        MeanhistvaluesLABRegionHighIntensityForest = []
        MeanhistvaluesLABRegionTotalIntensityForest = []
        MeanhistvaluesLABRegionLowIntensityStreet = []
        MeanhistvaluesLABRegionMidIntensityStreet = []
        MeanhistvaluesLABRegionHighIntensityStreet = []
        MeanhistvaluesLABRegionTotalIntensityStreet = []
        VarhistvaluesLABRegionLowIntensityCoast = []
        VarhistvaluesLABRegionMidIntensityCoast = []
        VarhistvaluesLABRegionHighIntensityCoast = []
        VarhistvaluesLABRegionTotalIntensityCoast = []
        VarhistvaluesLABRegionLowIntensityForest = []
        VarhistvaluesLABRegionMidIntensityForest = []
        VarhistvaluesLABRegionHighIntensityForest = []
        VarhistvaluesLABRegionTotalIntensityForest = []
        VarhistvaluesLABRegionLowIntensityStreet = []
        VarhistvaluesLABRegionMidIntensityStreet = []
        VarhistvaluesLABRegionHighIntensityStreet = []
        VarhistvaluesLABRegionTotalIntensityStreet = []
        i = 0
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins)  # External rescale pour Lab
            #imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            histvaluesLAB = self.generateHistogram(imageLabhist)
            #histvaluesHSV = self.generateHistogram(imageHSVhist)

            mean_r_low = (np.mean(histvaluesLAB[0][0:85]))
            mean_g_low = (np.mean(histvaluesLAB[1][0:85]))
            mean_b_low = (np.mean(histvaluesLAB[2][0:85]))
            mean_r_mid = (np.mean(histvaluesLAB[0][85:170]))
            mean_g_mid = (np.mean(histvaluesLAB[1][85:170]))
            mean_b_mid = (np.mean(histvaluesLAB[2][85:170]))
            mean_r_high = (np.mean(histvaluesLAB[0][170:255]))
            mean_g_high = (np.mean(histvaluesLAB[1][170:255]))
            mean_b_high = (np.mean(histvaluesLAB[2][170:255]))
            std_r_low = (np.var(histvaluesLAB[0][0:85]))
            std_g_low = (np.var(histvaluesLAB[1][0:85]))
            std_b_low = (np.var(histvaluesLAB[2][0:85]))
            std_r_mid = (np.var(histvaluesLAB[0][85:170]))
            std_g_mid = (np.var(histvaluesLAB[1][85:170]))
            std_b_mid = (np.var(histvaluesLAB[2][85:170]))
            std_r_high = (np.var(histvaluesLAB[0][170:255]))
            std_g_high = (np.var(histvaluesLAB[1][170:255]))
            std_b_high = (np.var(histvaluesLAB[2][170:255]))

            if "coast" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityCoast.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesLABRegionMidIntensityCoast.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesLABRegionHighIntensityCoast.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesLABRegionTotalIntensityCoast.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesLABRegionLowIntensityCoast.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesLABRegionMidIntensityCoast.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesLABRegionHighIntensityCoast.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesLABRegionTotalIntensityCoast.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            if "forest" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityForest.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesLABRegionMidIntensityForest.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesLABRegionHighIntensityForest.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesLABRegionTotalIntensityForest.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesLABRegionLowIntensityForest.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesLABRegionMidIntensityForest.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesLABRegionHighIntensityForest.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesLABRegionTotalIntensityForest.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            if "street" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityStreet.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesLABRegionMidIntensityStreet.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesLABRegionHighIntensityStreet.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesLABRegionTotalIntensityStreet.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesLABRegionLowIntensityStreet.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesLABRegionMidIntensityStreet.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesLABRegionHighIntensityStreet.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesLABRegionTotalIntensityStreet.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            i = i + 1
            print(i)

        avgMeanhistvaluesLABRegionLowIntensityCoast = np.mean(MeanhistvaluesLABRegionLowIntensityCoast, axis=0)
        avgMeanhistvaluesLABRegionMidIntensityCoast = np.mean(MeanhistvaluesLABRegionMidIntensityCoast, axis=0)
        avgMeanhistvaluesLABRegionHighIntensityCoast = np.mean(MeanhistvaluesLABRegionHighIntensityCoast, axis=0)
        avgMeanhistvaluesLABRegionTotalIntensityCoast = np.mean(MeanhistvaluesLABRegionTotalIntensityCoast, axis=0)
        avgVarhistvaluesLABRegionLowIntensityCoast = np.mean(VarhistvaluesLABRegionLowIntensityCoast, axis=0)
        avgVarhistvaluesLABRegionMidIntensityCoast = np.mean(VarhistvaluesLABRegionMidIntensityCoast, axis=0)
        avgVarhistvaluesLABRegionHighIntensityCoast = np.mean(VarhistvaluesLABRegionHighIntensityCoast, axis=0)
        avgVarhistvaluesLABRegionTotalIntensityCoast = np.mean(VarhistvaluesLABRegionTotalIntensityCoast, axis=0)

        avgMeanhistvaluesLABRegionLowIntensityForest = np.mean(MeanhistvaluesLABRegionLowIntensityForest, axis=0)
        avgMeanhistvaluesLABRegionMidIntensityForest = np.mean(MeanhistvaluesLABRegionMidIntensityForest, axis=0)
        avgMeanhistvaluesLABRegionHighIntensityForest = np.mean(MeanhistvaluesLABRegionHighIntensityForest, axis=0)
        avgMeanhistvaluesLABRegionTotalIntensityForest = np.mean(MeanhistvaluesLABRegionTotalIntensityForest, axis=0)
        avgVarhistvaluesLABRegionLowIntensityForest = np.mean(VarhistvaluesLABRegionLowIntensityForest, axis=0)
        avgVarhistvaluesLABRegionMidIntensityForest = np.mean(VarhistvaluesLABRegionMidIntensityForest, axis=0)
        avgVarhistvaluesLABRegionHighIntensityForest = np.mean(VarhistvaluesLABRegionHighIntensityForest, axis=0)
        avgVarhistvaluesLABRegionTotalIntensityForest = np.mean(VarhistvaluesLABRegionTotalIntensityForest, axis=0)

        avgMeanhistvaluesLABRegionLowIntensityStreet = np.mean(MeanhistvaluesLABRegionLowIntensityStreet, axis=0)
        avgMeanhistvaluesLABRegionMidIntensityStreet = np.mean(MeanhistvaluesLABRegionMidIntensityStreet, axis=0)
        avgMeanhistvaluesLABRegionHighIntensityStreet = np.mean(MeanhistvaluesLABRegionHighIntensityStreet, axis=0)
        avgMeanhistvaluesLABRegionTotalIntensityStreet = np.mean(MeanhistvaluesLABRegionTotalIntensityStreet, axis=0)
        avgVarhistvaluesLABRegionLowIntensityStreet = np.mean(VarhistvaluesLABRegionLowIntensityStreet, axis=0)
        avgVarhistvaluesLABRegionMidIntensityStreet = np.mean(VarhistvaluesLABRegionMidIntensityStreet, axis=0)
        avgVarhistvaluesLABRegionHighIntensityStreet = np.mean(VarhistvaluesLABRegionHighIntensityStreet, axis=0)
        avgVarhistvaluesLABRegionTotalIntensityStreet = np.mean(VarhistvaluesLABRegionTotalIntensityStreet, axis=0)

        print("avgMeanhistvaluesLABRegionLowIntensityCoast")
        print(avgMeanhistvaluesLABRegionLowIntensityCoast)
        print("avgMeanhistvaluesLABRegionMidIntensityCoast")
        print(avgMeanhistvaluesLABRegionMidIntensityCoast)
        print("avgMeanhistvaluesLABRegionHighIntensityCoast")
        print(avgMeanhistvaluesLABRegionHighIntensityCoast)
        print("avgMeanhistvaluesLABRegionTotalIntensityCoast")
        print(avgMeanhistvaluesLABRegionTotalIntensityCoast)
        print("avgVarhistvaluesLABRegionLowIntensityCoast")
        print(avgVarhistvaluesLABRegionLowIntensityCoast)
        print("avgVarhistvaluesLABRegionMidIntensityCoast")
        print(avgVarhistvaluesLABRegionMidIntensityCoast)
        print("avgVarhistvaluesLABRegionHighIntensityCoast")
        print(avgVarhistvaluesLABRegionHighIntensityCoast)
        print("avgVarhistvaluesLABRegionTotalIntensityCoast")
        print(avgVarhistvaluesLABRegionTotalIntensityCoast)

        print("avgMeanhistvaluesLABRegionLowIntensityForest")
        print(avgMeanhistvaluesLABRegionLowIntensityForest)
        print("avgMeanhistvaluesLABRegionMidIntensityForest")
        print(avgMeanhistvaluesLABRegionMidIntensityForest)
        print("avgMeanhistvaluesLABRegionHighIntensityForest")
        print(avgMeanhistvaluesLABRegionHighIntensityForest)
        print("avgMeanhistvaluesLABRegionTotalIntensityForest")
        print(avgMeanhistvaluesLABRegionTotalIntensityForest)
        print("avgVarhistvaluesLABRegionLowIntensityForest")
        print(avgVarhistvaluesLABRegionLowIntensityForest)
        print("avgVarhistvaluesLABRegionMidIntensityForest")
        print(avgVarhistvaluesLABRegionMidIntensityForest)
        print("avgVarhistvaluesLABRegionHighIntensityForest")
        print(avgVarhistvaluesLABRegionHighIntensityForest)
        print("avgVarhistvaluesLABRegionTotalIntensityForest")
        print(avgVarhistvaluesLABRegionTotalIntensityForest)

        print("avgMeanhistvaluesLABRegionLowIntensityStreet")
        print(avgMeanhistvaluesLABRegionLowIntensityStreet)
        print("avgMeanhistvaluesLABRegionMidIntensityStreet")
        print(avgMeanhistvaluesLABRegionMidIntensityStreet)
        print("avgMeanhistvaluesLABRegionHighIntensityStreet")
        print(avgMeanhistvaluesLABRegionHighIntensityStreet)
        print("avgMeanhistvaluesLABRegionTotalIntensityStreet")
        print(avgMeanhistvaluesLABRegionTotalIntensityStreet)
        print("avgVarhistvaluesLABRegionLowIntensityStreet")
        print(avgVarhistvaluesLABRegionLowIntensityStreet)
        print("avgVarhistvaluesLABRegionMidIntensityStreet")
        print(avgVarhistvaluesLABRegionMidIntensityStreet)
        print("avgVarhistvaluesLABRegionHighIntensityStreet")
        print(avgVarhistvaluesLABRegionHighIntensityStreet)
        print("avgVarhistvaluesLABRegionTotalIntensityStreet")
        print(avgVarhistvaluesLABRegionTotalIntensityStreet)

        return avgVarhistvaluesLABRegionHighIntensityStreet

    def histAvgRegionGenHSV(self):
        MeanhistvaluesLABRegionLowIntensityCoast = []

        MeanhistvaluesLABRegionLowIntensityForest = []

        MeanhistvaluesLABRegionLowIntensityStreet = []

        i = 0
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # Exemple de conversion de format pour Lab et HSV
            #imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            #imageLabhist = an.rescaleHistLab(imageLab, n_bins)  # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            #histvaluesLAB = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            std_r_low = (np.var(histvaluesHSV[0][85:170]))
            std_g_low = (np.var(histvaluesHSV[1][85:170]))
            std_b_low = (np.var(histvaluesHSV[2][85:170]))

            if "coast" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityCoast.append([std_r_low, std_g_low, std_b_low])

            if "forest" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityForest.append([std_r_low, std_g_low, std_b_low])

            if "street" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityStreet.append([std_r_low, std_g_low, std_b_low])
            i = i + 1
            print(i)

        return MeanhistvaluesLABRegionLowIntensityCoast + MeanhistvaluesLABRegionLowIntensityForest + MeanhistvaluesLABRegionLowIntensityStreet

    def histAvgRegionGenLAB(self):
        MeanhistvaluesLABRegionLowIntensityCoast = []

        MeanhistvaluesLABRegionLowIntensityForest = []

        MeanhistvaluesLABRegionLowIntensityStreet = []

        i = 0
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins)  # External rescale pour Lab
            #imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            histvaluesLAB = self.generateHistogram(imageLabhist)
            #histvaluesHSV = self.generateHistogram(imageHSVhist)

            std_r_low = (np.mean(histvaluesLAB[0][0:85]))
            std_g_low = (np.mean(histvaluesLAB[1][0:85]))
            std_b_low = (np.mean(histvaluesLAB[2][0:85]))


            var_r_low = (np.mean(histvaluesLAB[0][170:255]))
            var_g_low = (np.mean(histvaluesLAB[1][170:255]))
            var_b_low = (np.mean(histvaluesLAB[2][170:255]))
            if "coast" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityCoast.append([std_r_low, std_g_low, std_b_low])

            if "forest" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityForest.append([std_r_low, std_g_low, std_b_low])

            if "street" in self.image_list[image_counter]:
                MeanhistvaluesLABRegionLowIntensityStreet.append([std_r_low, std_g_low, std_b_low])
            i = i + 1
            print(i)

        return MeanhistvaluesLABRegionLowIntensityCoast + MeanhistvaluesLABRegionLowIntensityForest + MeanhistvaluesLABRegionLowIntensityStreet

    def histAvgRegionGenRGB(self):

        VarhistvaluesRGBRegionHighIntensityCoast = []
        VarhistvaluesRGBRegionHighIntensityForest = []
        VarhistvaluesRGBRegionHighIntensityStreet = []
        i = 0
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # Exemple de conversion de format pour Lab et HSV

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            std_r_high = (np.mean(histvaluesRGB[0][170:255]))
            std_g_high = (np.mean(histvaluesRGB[1][170:255]))
            std_b_high = (np.mean(histvaluesRGB[2][170:255]))

            if "coast" in self.image_list[image_counter]:
                VarhistvaluesRGBRegionHighIntensityCoast.append([std_r_high,std_g_high,std_b_high])
            if "forest" in self.image_list[image_counter]:
                VarhistvaluesRGBRegionHighIntensityForest.append([std_r_high, std_g_high, std_b_high])
            if "street" in self.image_list[image_counter]:
                VarhistvaluesRGBRegionHighIntensityStreet.append([std_r_high, std_g_high, std_b_high])
            i = i + 1
            print(i)

        return VarhistvaluesRGBRegionHighIntensityCoast + VarhistvaluesRGBRegionHighIntensityForest + VarhistvaluesRGBRegionHighIntensityStreet

    def histAvgRegionAnalysisRGB(self):
        MeanhistvaluesRGBRegionLowIntensityCoast = []
        MeanhistvaluesRGBRegionMidIntensityCoast = []
        MeanhistvaluesRGBRegionHighIntensityCoast = []
        MeanhistvaluesRGBRegionTotalIntensityCoast = []
        MeanhistvaluesRGBRegionLowIntensityForest = []
        MeanhistvaluesRGBRegionMidIntensityForest = []
        MeanhistvaluesRGBRegionHighIntensityForest = []
        MeanhistvaluesRGBRegionTotalIntensityForest = []
        MeanhistvaluesRGBRegionLowIntensityStreet = []
        MeanhistvaluesRGBRegionMidIntensityStreet = []
        MeanhistvaluesRGBRegionHighIntensityStreet = []
        MeanhistvaluesRGBRegionTotalIntensityStreet = []
        VarhistvaluesRGBRegionLowIntensityCoast = []
        VarhistvaluesRGBRegionMidIntensityCoast = []
        VarhistvaluesRGBRegionHighIntensityCoast = []
        VarhistvaluesRGBRegionTotalIntensityCoast = []
        VarhistvaluesRGBRegionLowIntensityForest = []
        VarhistvaluesRGBRegionMidIntensityForest = []
        VarhistvaluesRGBRegionHighIntensityForest = []
        VarhistvaluesRGBRegionTotalIntensityForest = []
        VarhistvaluesRGBRegionLowIntensityStreet = []
        VarhistvaluesRGBRegionMidIntensityStreet = []
        VarhistvaluesRGBRegionHighIntensityStreet = []
        VarhistvaluesRGBRegionTotalIntensityStreet = []
        i = 0
        for image_counter in range(len(self.image_list)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            # Exemple de conversion de format pour Lab et HSV

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)

            mean_r_low = (np.mean(histvaluesRGB[0][0:85]))
            mean_g_low = (np.mean(histvaluesRGB[1][0:85]))
            mean_b_low = (np.mean(histvaluesRGB[2][0:85]))
            mean_r_mid = (np.mean(histvaluesRGB[0][85:170]))
            mean_g_mid = (np.mean(histvaluesRGB[1][85:170]))
            mean_b_mid = (np.mean(histvaluesRGB[2][85:170]))
            mean_r_high = (np.mean(histvaluesRGB[0][170:255]))
            mean_g_high = (np.mean(histvaluesRGB[1][170:255]))
            mean_b_high = (np.mean(histvaluesRGB[2][170:255]))
            std_r_low = (np.var(histvaluesRGB[0][0:85]))
            std_g_low = (np.var(histvaluesRGB[1][0:85]))
            std_b_low = (np.var(histvaluesRGB[2][0:85]))
            std_r_mid = (np.var(histvaluesRGB[0][85:170]))
            std_g_mid = (np.var(histvaluesRGB[1][85:170]))
            std_b_mid = (np.var(histvaluesRGB[2][85:170]))
            std_r_high = (np.var(histvaluesRGB[0][170:255]))
            std_g_high = (np.var(histvaluesRGB[1][170:255]))
            std_b_high = (np.var(histvaluesRGB[2][170:255]))

            if "coast" in self.image_list[image_counter]:
                MeanhistvaluesRGBRegionLowIntensityCoast.append([mean_r_low,mean_g_low,mean_b_low])
                MeanhistvaluesRGBRegionMidIntensityCoast.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesRGBRegionHighIntensityCoast.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesRGBRegionTotalIntensityCoast.append([(mean_r_low+mean_r_mid+mean_r_high)/3, (mean_g_low+mean_g_mid+mean_g_high)/3, (mean_b_low+mean_b_mid+mean_b_high)/3])
                VarhistvaluesRGBRegionLowIntensityCoast.append([std_r_low,std_g_low,std_b_low])
                VarhistvaluesRGBRegionMidIntensityCoast.append([std_r_mid,std_g_mid,std_b_mid])
                VarhistvaluesRGBRegionHighIntensityCoast.append([std_r_high,std_g_high,std_b_high])
                VarhistvaluesRGBRegionTotalIntensityCoast.append([(std_r_low+std_r_mid+std_r_high)/3, (std_g_low+std_g_mid+std_g_high)/3, (std_b_low+std_b_mid+std_b_high)/3])
            if "forest" in self.image_list[image_counter]:
                MeanhistvaluesRGBRegionLowIntensityForest.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesRGBRegionMidIntensityForest.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesRGBRegionHighIntensityForest.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesRGBRegionTotalIntensityForest.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesRGBRegionLowIntensityForest.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesRGBRegionMidIntensityForest.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesRGBRegionHighIntensityForest.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesRGBRegionTotalIntensityForest.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            if "street" in self.image_list[image_counter]:
                MeanhistvaluesRGBRegionLowIntensityStreet.append([mean_r_low, mean_g_low, mean_b_low])
                MeanhistvaluesRGBRegionMidIntensityStreet.append([mean_r_mid, mean_g_mid, mean_b_mid])
                MeanhistvaluesRGBRegionHighIntensityStreet.append([mean_r_high, mean_g_high, mean_b_high])
                MeanhistvaluesRGBRegionTotalIntensityStreet.append(
                    [(mean_r_low + mean_r_mid + mean_r_high) / 3, (mean_g_low + mean_g_mid + mean_g_high) / 3,
                     (mean_b_low + mean_b_mid + mean_b_high) / 3])
                VarhistvaluesRGBRegionLowIntensityStreet.append([std_r_low, std_g_low, std_b_low])
                VarhistvaluesRGBRegionMidIntensityStreet.append([std_r_mid, std_g_mid, std_b_mid])
                VarhistvaluesRGBRegionHighIntensityStreet.append([std_r_high, std_g_high, std_b_high])
                VarhistvaluesRGBRegionTotalIntensityStreet.append(
                    [(std_r_low + std_r_mid + std_r_high) / 3, (std_g_low + std_g_mid + std_g_high) / 3,
                     (std_b_low + std_b_mid + std_b_high) / 3])
            i = i + 1
            print(i)

        avgMeanhistvaluesRGBRegionLowIntensityCoast = np.mean(MeanhistvaluesRGBRegionLowIntensityCoast, axis=0)
        avgMeanhistvaluesRGBRegionMidIntensityCoast = np.mean(MeanhistvaluesRGBRegionMidIntensityCoast, axis=0)
        avgMeanhistvaluesRGBRegionHighIntensityCoast = np.mean(MeanhistvaluesRGBRegionHighIntensityCoast, axis=0)
        avgMeanhistvaluesRGBRegionTotalIntensityCoast = np.mean(MeanhistvaluesRGBRegionTotalIntensityCoast, axis=0)
        avgVarhistvaluesRGBRegionLowIntensityCoast = np.mean(VarhistvaluesRGBRegionLowIntensityCoast, axis=0)
        avgVarhistvaluesRGBRegionMidIntensityCoast = np.mean(VarhistvaluesRGBRegionMidIntensityCoast, axis=0)
        avgVarhistvaluesRGBRegionHighIntensityCoast = np.mean(VarhistvaluesRGBRegionHighIntensityCoast, axis=0)
        avgVarhistvaluesRGBRegionTotalIntensityCoast = np.mean(VarhistvaluesRGBRegionTotalIntensityCoast, axis=0)

        avgMeanhistvaluesRGBRegionLowIntensityForest = np.mean(MeanhistvaluesRGBRegionLowIntensityForest, axis=0)
        avgMeanhistvaluesRGBRegionMidIntensityForest = np.mean(MeanhistvaluesRGBRegionMidIntensityForest, axis=0)
        avgMeanhistvaluesRGBRegionHighIntensityForest = np.mean(MeanhistvaluesRGBRegionHighIntensityForest , axis=0)
        avgMeanhistvaluesRGBRegionTotalIntensityForest = np.mean(MeanhistvaluesRGBRegionTotalIntensityForest, axis=0)
        avgVarhistvaluesRGBRegionLowIntensityForest = np.mean(VarhistvaluesRGBRegionLowIntensityForest, axis=0)
        avgVarhistvaluesRGBRegionMidIntensityForest = np.mean(VarhistvaluesRGBRegionMidIntensityForest , axis=0)
        avgVarhistvaluesRGBRegionHighIntensityForest = np.mean(VarhistvaluesRGBRegionHighIntensityForest, axis=0)
        avgVarhistvaluesRGBRegionTotalIntensityForest = np.mean(VarhistvaluesRGBRegionTotalIntensityForest, axis=0)

        avgMeanhistvaluesRGBRegionLowIntensityStreet = np.mean(MeanhistvaluesRGBRegionLowIntensityStreet, axis=0)
        avgMeanhistvaluesRGBRegionMidIntensityStreet = np.mean(MeanhistvaluesRGBRegionMidIntensityStreet, axis=0)
        avgMeanhistvaluesRGBRegionHighIntensityStreet = np.mean(MeanhistvaluesRGBRegionHighIntensityStreet, axis=0)
        avgMeanhistvaluesRGBRegionTotalIntensityStreet = np.mean(MeanhistvaluesRGBRegionTotalIntensityStreet, axis=0)
        avgVarhistvaluesRGBRegionLowIntensityStreet = np.mean(VarhistvaluesRGBRegionLowIntensityStreet, axis=0)
        avgVarhistvaluesRGBRegionMidIntensityStreet = np.mean(VarhistvaluesRGBRegionMidIntensityStreet, axis=0)
        avgVarhistvaluesRGBRegionHighIntensityStreet = np.mean(VarhistvaluesRGBRegionHighIntensityStreet , axis=0)
        avgVarhistvaluesRGBRegionTotalIntensityStreet = np.mean(VarhistvaluesRGBRegionTotalIntensityStreet, axis=0)

        print("avgMeanhistvaluesRGBRegionLowIntensityCoast")
        print(avgMeanhistvaluesRGBRegionLowIntensityCoast)
        print("avgMeanhistvaluesRGBRegionMidIntensityCoast")
        print(avgMeanhistvaluesRGBRegionMidIntensityCoast)
        print("avgMeanhistvaluesRGBRegionHighIntensityCoast")
        print(avgMeanhistvaluesRGBRegionHighIntensityCoast )
        print("avgMeanhistvaluesRGBRegionTotalIntensityCoast")
        print(avgMeanhistvaluesRGBRegionTotalIntensityCoast)
        print("avgVarhistvaluesRGBRegionLowIntensityCoast")
        print(avgVarhistvaluesRGBRegionLowIntensityCoast )
        print("avgVarhistvaluesRGBRegionMidIntensityCoast")
        print(avgVarhistvaluesRGBRegionMidIntensityCoast )
        print("avgVarhistvaluesRGBRegionHighIntensityCoast")
        print(avgVarhistvaluesRGBRegionHighIntensityCoast)
        print("avgVarhistvaluesRGBRegionTotalIntensityCoast")
        print(avgVarhistvaluesRGBRegionTotalIntensityCoast )

        print("avgMeanhistvaluesRGBRegionLowIntensityForest")
        print(avgMeanhistvaluesRGBRegionLowIntensityForest)
        print("avgMeanhistvaluesRGBRegionMidIntensityForest")
        print(avgMeanhistvaluesRGBRegionMidIntensityForest)
        print("avgMeanhistvaluesRGBRegionHighIntensityForest")
        print(avgMeanhistvaluesRGBRegionHighIntensityForest)
        print("avgMeanhistvaluesRGBRegionTotalIntensityForest")
        print(avgMeanhistvaluesRGBRegionTotalIntensityForest)
        print("avgVarhistvaluesRGBRegionLowIntensityForest" )
        print(avgVarhistvaluesRGBRegionLowIntensityForest )
        print("avgVarhistvaluesRGBRegionMidIntensityForest" )
        print(avgVarhistvaluesRGBRegionMidIntensityForest )
        print("avgVarhistvaluesRGBRegionHighIntensityForest" )
        print(avgVarhistvaluesRGBRegionHighIntensityForest )
        print("avgVarhistvaluesRGBRegionTotalIntensityForest")
        print(avgVarhistvaluesRGBRegionTotalIntensityForest)

        print("avgMeanhistvaluesRGBRegionLowIntensityStreet")
        print(avgMeanhistvaluesRGBRegionLowIntensityStreet)
        print("avgMeanhistvaluesRGBRegionMidIntensityStreet")
        print(avgMeanhistvaluesRGBRegionMidIntensityStreet)
        print("avgMeanhistvaluesRGBRegionHighIntensityStreet")
        print(avgMeanhistvaluesRGBRegionHighIntensityStreet)
        print("avgMeanhistvaluesRGBRegionTotalIntensityStreet")
        print(avgMeanhistvaluesRGBRegionTotalIntensityStreet)
        print("avgVarhistvaluesRGBRegionLowIntensityStreet")
        print(avgVarhistvaluesRGBRegionLowIntensityStreet )
        print("avgVarhistvaluesRGBRegionMidIntensityStreet" )
        print(avgVarhistvaluesRGBRegionMidIntensityStreet )
        print("avgVarhistvaluesRGBRegionHighIntensityStreet")
        print(avgVarhistvaluesRGBRegionHighIntensityStreet)
        print("avgVarhistvaluesRGBRegionTotalIntensityStreet")
        print(avgVarhistvaluesRGBRegionTotalIntensityStreet)

        return avgVarhistvaluesRGBRegionHighIntensityStreet
    def colorAvgAnalysis(self):
        avg_mean_rgb_coast = []
        avg_std_rgb_coast = []
        avg_mean_rgb_forest = []
        avg_std_rgb_forest = []
        avg_mean_rgb_street = []
        avg_std_rgb_street = []
        for image_counter in range(len(self.image_list)):
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])
            if "coast" in self.image_list[image_counter]:
                avg_mean_rgb_coast.append(np.mean(imageRGB, axis = (0,1)))
                avg_std_rgb_coast.append(np.var(imageRGB, axis = (0,1)))
            if "forest" in self.image_list[image_counter]:
                avg_mean_rgb_forest.append(np.mean(imageRGB, axis = (0,1)))
                avg_std_rgb_forest.append(np.var(imageRGB, axis = (0,1)))
            if "street" in self.image_list[image_counter]:
                avg_mean_rgb_street.append(np.mean(imageRGB, axis = (0,1)))
                avg_std_rgb_street.append(np.var(imageRGB, axis = (0,1)))

        avg_mean_rgb_coast_of_all_images = np.mean(avg_mean_rgb_coast, axis=0)
        avg_std_rgb_coast_of_all_images = np.mean(avg_std_rgb_coast, axis=0)
        avg_mean_rgb_forest_of_all_images = np.mean(avg_mean_rgb_forest, axis=0)
        avg_std_rgb_forest_of_all_images = np.mean(avg_std_rgb_forest, axis=0)
        avg_mean_rgb_street_of_all_images = np.mean(avg_mean_rgb_street, axis=0)
        avg_std_rgb_street_of_all_images = np.mean(avg_std_rgb_street, axis=0)

        print("the average mean of the RGB for the coasts")
        print(avg_mean_rgb_coast_of_all_images)
        print("the average var of the RGB for the coasts")
        print(avg_std_rgb_coast_of_all_images)
        print("the average var of the RGB for the coasts")
        print(avg_mean_rgb_forest_of_all_images)
        print("the average var of the RGB for the coasts")
        print(avg_std_rgb_forest_of_all_images)
        print("the average var of the RGB for the coasts")
        print(avg_mean_rgb_street_of_all_images)
        print("the average var of the RGB for the coasts")
        print(avg_std_rgb_street_of_all_images)

        avg_mean_lab_coast = []
        avg_std_lab_coast = []
        avg_mean_lab_forest = []
        avg_std_lab_forest = []
        avg_mean_lab_street = []
        avg_std_lab_street = []
        for image_counter in range(len(self.image_list)):
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])
            imageLab = skic.rgb2lab(imageRGB)
            if "coast" in self.image_list[image_counter]:
                avg_mean_lab_coast.append(np.mean(imageLab, axis=(0, 1)))
                avg_std_lab_coast.append(np.var(imageLab, axis=(0, 1)))
            if "forest" in self.image_list[image_counter]:
                avg_mean_lab_forest.append(np.mean(imageLab, axis=(0, 1)))
                avg_std_lab_forest.append(np.var(imageLab, axis=(0, 1)))
            if "street" in self.image_list[image_counter]:
                avg_mean_lab_street.append(np.mean(imageLab, axis=(0, 1)))
                avg_std_lab_street.append(np.var(imageLab, axis=(0, 1)))

        avg_mean_lab_coast_of_all_images = np.mean(avg_mean_lab_coast, axis=0)
        avg_std_lab_coast_of_all_images = np.mean(avg_std_lab_coast, axis=0)
        avg_mean_lab_forest_of_all_images = np.mean(avg_mean_lab_forest, axis=0)
        avg_std_lab_forest_of_all_images = np.mean(avg_std_lab_forest, axis=0)
        avg_mean_lab_street_of_all_images = np.mean(avg_mean_lab_street, axis=0)
        avg_std_lab_street_of_all_images = np.mean(avg_std_lab_street, axis=0)

        print(avg_mean_lab_coast_of_all_images)
        print(avg_std_lab_coast_of_all_images)
        print(avg_mean_lab_forest_of_all_images)
        print(avg_std_lab_forest_of_all_images)
        print(avg_mean_lab_street_of_all_images)
        print(avg_std_lab_street_of_all_images)

        avg_mean_hsv_coast = []
        avg_std_hsv_coast = []
        avg_mean_hsv_forest = []
        avg_std_hsv_forest = []
        avg_mean_hsv_street = []
        avg_std_hsv_street = []
        for image_counter in range(len(self.image_list)):
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[image_counter])

            imagehsv = skic.rgb2hsv(imageRGB)
            if "coast" in self.image_list[image_counter]:
                avg_mean_hsv_coast.append(np.mean(imagehsv, axis=(0, 1)))
                avg_std_hsv_coast.append(np.var(imagehsv, axis=(0, 1)))
            if "forest" in self.image_list[image_counter]:
                avg_mean_hsv_forest.append(np.mean(imagehsv, axis=(0, 1)))
                avg_std_hsv_forest.append(np.var(imagehsv, axis=(0, 1)))
            if "street" in self.image_list[image_counter]:
                avg_mean_hsv_street.append(np.mean(imagehsv, axis=(0, 1)))
                avg_std_hsv_street.append(np.var(imagehsv, axis=(0, 1)))

        avg_mean_hsv_coast_of_all_images = np.mean(avg_mean_hsv_coast, axis=0)
        avg_std_hsv_coast_of_all_images = np.mean(avg_std_hsv_coast, axis=0)
        avg_mean_hsv_forest_of_all_images = np.mean(avg_mean_hsv_forest, axis=0)
        avg_std_hsv_forest_of_all_images = np.mean(avg_std_hsv_forest, axis=0)
        avg_mean_hsv_street_of_all_images = np.mean(avg_mean_hsv_street, axis=0)
        avg_std_hsv_street_of_all_images = np.mean(avg_std_hsv_street, axis=0)

        print(avg_mean_hsv_coast_of_all_images)
        print(avg_std_hsv_coast_of_all_images)
        print(avg_mean_hsv_forest_of_all_images)
        print(avg_std_hsv_forest_of_all_images)
        print(avg_mean_hsv_street_of_all_images)
        print(avg_std_hsv_street_of_all_images)


        representationOfImages = []
        for i in range(0,len(avg_mean_rgb_coast)):
            B = avg_mean_rgb_coast[i][2]
            VarRGB = (avg_std_rgb_coast[i][0] + avg_std_rgb_coast[i][1] + avg_std_rgb_coast[i][2])/3
            VarL = avg_mean_lab_coast[i][2]
            representationOfImages.append([B,VarRGB,VarL])
        for i in range(0,len(avg_mean_rgb_forest)):
            B = avg_mean_rgb_forest[i][2]
            VarRGB = (avg_std_rgb_forest[i][0] + avg_std_rgb_forest[i][1] + avg_std_rgb_forest[i][2])/3
            VarL = avg_mean_lab_forest[i][2]
            representationOfImages.append([B,VarRGB,VarL])
        for i in range(0,len(avg_mean_rgb_street)):
            B = avg_mean_rgb_street[i][2]
            VarRGB = (avg_std_rgb_street[i][0] + avg_std_rgb_street[i][1] + avg_std_rgb_street[i][2])/3
            VarL = avg_mean_lab_street[i][2]
            representationOfImages.append([B,VarRGB,VarL])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sizes = [50] * len(representationOfImages)
        colors = ['red' if i < 328 else 'blue' if 328 <= i < 688 else 'green' for i in
                  range(len(representationOfImages))]

        # Extract coordinates for each axis
        x_coords = [point[0] for point in representationOfImages]
        y_coords = [point[1] for point in representationOfImages]
        z_coords = [point[2] for point in representationOfImages]

        # Create a 3D scatter plot with specified point sizes and colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points with their corresponding size and color
        ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes)

        # Set labels for each axis
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # Display the plot
        plt.show()




    def generateRepresentation(self):
        # produce a ClassificationData object usable by the classifiers
        # TODO L1.E4.8: commencer l'analyse de la représentation choisie

        raise NotImplementedError()

    def load_images(self, indexes):
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]
        im = []
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im.append(self.images[i])
            else:
                im.append(skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]]))
        return im

    def images_display(self, indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """
        im = self.load_images(indexes)
        fig2 = plt.figure()
        ax2 = fig2.subplots(len(im), 1)
        for i, v in enumerate(im):
            ax2[i].imshow(v)

    def view_histogrammes(self, indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for image_counter in range(len(indexes)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            # affichage des histogrammes
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c='red')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c='green')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c='blue')
            ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 1].set_title(f'histogramme LAB de {image_name}')
            # TODO L1.E4 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots

            # 2e histogramme
            # affichage des histogrammes
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='red')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='green')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c='blue')
            ax[image_counter, 2].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 2].set_title(f'histogramme HSV de {image_name}')
            # TODO L1.E4 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
