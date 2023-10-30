import numpy as np
from helpers.ImageCollection import ImageCollection
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_collection = ImageCollection()
    edge_coef = -0.5
    k1 = np.array([[edge_coef, 1, edge_coef],
                   [edge_coef, 1, edge_coef],
                   [edge_coef, 1, edge_coef]])
    k2 = np.array([[edge_coef, edge_coef, edge_coef],
                   [1, 1, 1],
                   [edge_coef, edge_coef, edge_coef]])
    k3 = np.array([[1, edge_coef, edge_coef],
                   [edge_coef, 1, edge_coef],
                   [edge_coef, edge_coef, 1]])
    k4 = np.array([[edge_coef, edge_coef, 1],
                   [edge_coef, 1, edge_coef],
                   [1, edge_coef, edge_coef]])

    #Original convolution test made by William
    if False:
        N_samples = 5
        images_indexes = image_collection.get_samples(N_samples)
        images = image_collection.load_images(images_indexes)
        for i, image in enumerate(images):
            image_normalized = normalize_hist_rgb(image)
            imageHSV = skic.rgb2hsv(image)
            image_filtered = imageHSV
            vert_filter = cv2.filter2D(src=image_filtered, ddepth=-1, kernel=k1)
            horiz_filter = cv2.filter2D(src=image_filtered, ddepth=-1, kernel=k2)
            back_diag_filter = cv2.filter2D(src=image_filtered, ddepth=-1, kernel=k3)
            forward_diag_filter = cv2.filter2D(src=image_filtered, ddepth=-1, kernel=k4)
            res = np.hstack((image, image_filtered, horiz_filter, back_diag_filter, forward_diag_filter))
            cv2.imshow("original", res)
            cv2.waitKey()
            cv2.destroyAllWindows()
    #Checks if an average over RGB channels is better depending on the filtering of specific edges
    if False:
        images = image_collection.load_images(619)
        images_indexes = [i for i in range(620)]
        images = image_collection.load_images(images_indexes)
        sumRCoast = []
        sumGCoast = []
        sumBCoast = []
        sumRForest = []
        sumGForest = []
        sumBForest = []
        sumRStreet = []
        sumGStreet = []
        sumBStreet = []
        forestinstancesR = []
        forestinstancesG = []
        forestinstancesB = []
        streetinstancesR = []
        streetinstancesG = []
        streetinstancesB = []
        count = 0
        for r, image in enumerate(images):
            # image_normalized = normalize_hist_rgb(image)
            vert_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k1)
            horiz_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k2)
            back_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k3)
            forward_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k4)
            # image_filtered = cv2.bilateralFilter(image,9,75,75)
            forestSpecificInstanceR = []
            forestSpecificInstanceG = []
            forestSpecificInstanceB = []
            streetSpecificInstanceR = []
            streetSpecificInstanceG = []
            streetSpecificInstanceB = []
            for i in range(0, 255):
                for j in range(0, 255):
                    if np.sum(vert_filter[i][j]) == 0 and np.sum(back_diag_filter[i][j]) == 0 and np.sum(
                            horiz_filter[i][j]) == 0 and np.sum(vert_filter[i][j]) == 0:
                        if "coast" in image_collection.image_list[r]:
                            sumRCoast.append(image[i][j][0])
                            sumGCoast.append(image[i][j][1])
                            sumBCoast.append(image[i][j][2])
                        if "forest" in image_collection.image_list[r]:
                            sumRForest.append(image[i][j][0])
                            sumGForest.append(image[i][j][1])
                            sumBForest.append(image[i][j][2])
                            forestSpecificInstanceR.append(image[i][j][0])
                            forestSpecificInstanceG.append(image[i][j][1])
                            forestSpecificInstanceB.append(image[i][j][2])
                        if "street" in image_collection.image_list[r]:
                            sumRStreet.append(image[i][j][0])
                            sumGStreet.append(image[i][j][1])
                            sumBStreet.append(image[i][j][2])
                            streetSpecificInstanceR.append(image[i][j][0])
                            streetSpecificInstanceG.append(image[i][j][1])
                            streetSpecificInstanceB.append(image[i][j][2])
            if "forest" in image_collection.image_list[r]:
                forestinstancesR.append(forestSpecificInstanceR)
                forestinstancesG.append(forestSpecificInstanceG)
                forestinstancesB.append(forestSpecificInstanceB)
            if "street" in image_collection.image_list[r]:
                streetinstancesR.append(streetSpecificInstanceR)
                streetinstancesG.append(streetSpecificInstanceG)
                streetinstancesB.append(streetSpecificInstanceB)
            count = count + 1
            print(count)

        # MeansumRCoast = np.mean(sumRCoast)
        # MeansumGCoast = np.mean(sumGCoast)
        # MeansumBCoast = np.mean(sumBCoast)
        MeansumRForest = np.mean(sumRForest)
        MeansumGForest = np.mean(sumGForest)
        MeansumBForest = np.mean(sumBForest)
        MeansumRStreet = np.mean(sumRStreet)
        MeansumGStreet = np.mean(sumGStreet)
        MeansumBStreet = np.mean(sumBStreet)
        #    print("Forest")
        #    print("R mean")
        #    for instance in forestinstancesR:

        #        print(np.mean(instance))
        #    print("G mean")
        #    for instance in forestinstancesG:

        #        print(np.mean(instance))
        #    print("B mean")
        errorforest = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 89:
                errorforest = errorforest + 1
        errorforest1 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 90:
                errorforest1 = errorforest1 + 1
        errorforest2 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 91:
                errorforest2 = errorforest2 + 1
        errorforest3 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 92:
                errorforest3 = errorforest3 + 1
        errorforest7 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 93:
                errorforest7 = errorforest7 + 1
        errorforest4 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 94:
                errorforest4 = errorforest4 + 1
        errorforest5 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 95:
                errorforest5 = errorforest5 + 1
        errorforest6 = 0
        for instance in forestinstancesB:
            if np.mean(instance) > 96:
                errorforest6 = errorforest6 + 1
        #    print("Ville")
        #    print("R mean")
        #    for instance in streetinstancesR:

        #        print(np.mean(instance))
        #    print("G mean")
        #    for instance in streetinstancesG:

        #        print(np.mean(instance))
        #    print("B mean")
        errorstreet = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 89:
                errorstreet = errorstreet + 1
        errorstreet1 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 90:
                errorstreet1 = errorstreet1 + 1
        errorstreet2 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 91:
                errorstreet2 = errorstreet2 + 1

        errorstreet3 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 92:
                errorstreet3 = errorstreet3 + 1

        errorstreet4 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 93:
                errorstreet4 = errorstreet4 + 1
        errorstreet5 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 94:
                errorstreet5 = errorstreet5 + 1
        errorstreet6 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 95:
                errorstreet6 = errorstreet6 + 1
        errorstreet7 = 0
        for instance in streetinstancesB:
            if np.mean(instance) < 96:
                errorstreet7 = errorstreet7 + 1

        print("errorforest")
        print(errorforest)
        print(errorforest1)
        print(errorforest2)
        print(errorforest3)
        print(errorforest4)
        print(errorforest5)
        print(errorforest6)
        print(errorforest7)
        print("errorstreet")
        print(errorstreet)
        print(errorstreet1)
        print(errorstreet2)
        print(errorstreet3)
        print(errorstreet4)
        print(errorstreet5)
        print(errorstreet6)
        print(errorstreet7)

        # print("MeansumRCoast")
        # print(MeansumRCoast)
        # print("MeansumGCoast")
        # print(MeansumGCoast)
        # print("MeansumBCoast")
        # print(MeansumBCoast)
        print("MeansumRForest")
        print(MeansumRForest)
        print("MeansumGForest")
        print(MeansumGForest)
        print("MeansumBForest")
        print(MeansumBForest)
        print("MeansumRStreet")
        print(MeansumRStreet)
        print("MeansumGStreet")
        print(MeansumGStreet)
        print("MeansumBStreet")
        print(MeansumBStreet)
    #Checks if the average pixel count over specific region on an RGB image is discriminatory
    if False:
        PixelCountCoastLow = []
        PixelCountCoastMid = []
        PixelCountCoastHigh = []
        PixelCountForestLow = []
        PixelCountForestMid = []
        PixelCountForestHigh = []
        PixelCountStreetLow = []
        PixelCountStreetMid = []
        PixelCountStreetHigh = []

        count = 0
        # images = image_collection.load_images(619)
        N_samples = 5
        # images_indexes = image_collection.get_samples(N_samples)
        images_indexes = [i for i in range(126)]
        images = image_collection.load_images(images_indexes)
        for r, image in enumerate(images):
            image_filtered = cv2.bilateralFilter(image, 18, 75, 75)
            imageLab = image_filtered
            vert_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k1)
            horiz_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k2)
            back_diag_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k3)
            forward_diag_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k4)
            # res = np.hstack((image, image_filtered, horiz_filter, back_diag_filter, forward_diag_filter))
            # cv2.imshow("original", res)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # image_filtered = cv2.bilateralFilter(image,9,75,75)
            pixelCountLow = 0
            pixelCountMid = 0
            pixelCountHigh = 0
            for i in range(0, 255):
                for j in range(85, 170):
                    if np.sum(back_diag_filter[i][j]) < 5 and np.sum(forward_diag_filter[i][j]) < 5 and np.sum(
                            vert_filter[i][j]) < 5 and np.sum(back_diag_filter[i][j]) < 5:
                        if "coast" in image_collection.image_list[r]:
                            pixelCountHigh = pixelCountHigh + 1

                        if "forest" in image_collection.image_list[r]:
                            pixelCountHigh = pixelCountHigh + 1
                        if "street" in image_collection.image_list[r]:
                            pixelCountHigh = pixelCountHigh + 1
            if "coast" in image_collection.image_list[r]:
                PixelCountCoastHigh.append(pixelCountHigh)
            if "forest" in image_collection.image_list[r]:
                PixelCountForestHigh.append(pixelCountHigh)
            if "street" in image_collection.image_list[r]:
                PixelCountStreetHigh.append(pixelCountHigh)

            count = count + 1
            print(count)
        CoastMeanHigh = np.mean(PixelCountCoastHigh)
        ForestMeanHigh = np.mean(PixelCountForestHigh)
        StreetMeanHigh = np.mean(PixelCountStreetHigh)
        #
        print("CoastMean")

        print(CoastMeanHigh)
        print("ForestMean")

        print(ForestMeanHigh)
        print("StreetMean")

        print(StreetMeanHigh)
    # creates .txt files that will be use by the ClassificationDataObject
    if True:
        images_indexes = [i for i in range(980)]
        images = image_collection.load_images(images_indexes)
        sumBCoast = []
        sumBForest = []
        sumBStreet = []
        PixelCountCoastHigh = []
        PixelCountForestHigh = []
        PixelCountStreetHigh = []
        count = 0
        for r, image in enumerate(images):
            image_filtered = cv2.bilateralFilter(image, 18, 75, 75)
            imageLab = image_filtered
            vert_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k1)
            horiz_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k2)
            back_diag_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k3)
            forward_diag_filter = cv2.filter2D(src=imageLab, ddepth=-1, kernel=k4)
            vert_filter1 = cv2.filter2D(src=image, ddepth=-1, kernel=k1)
            horiz_filter1 = cv2.filter2D(src=image, ddepth=-1, kernel=k2)
            back_diag_filter1 = cv2.filter2D(src=image, ddepth=-1, kernel=k3)
            forward_diag_filter1 = cv2.filter2D(src=image, ddepth=-1, kernel=k4)
            pixelCountHigh = 0
            sumB = []
            for i in range(0, 255):
                for j in range(0, 255):
                    if i >= 85 and i < 170 and np.sum(back_diag_filter[i][j]) < 5 and np.sum(
                            forward_diag_filter[i][j]) < 5 and np.sum(
                            vert_filter[i][j]) < 5 and np.sum(back_diag_filter[i][j]) < 5:
                        if "coast" in image_collection.image_list[r]:
                            pixelCountHigh = pixelCountHigh + 1
                        if "forest" in image_collection.image_list[r]:
                            pixelCountHigh = pixelCountHigh + 1
                        if "street" in image_collection.image_list[r]:
                            pixelCountHigh = pixelCountHigh + 1
                    if np.sum(vert_filter1[i][j]) == 0 and np.sum(back_diag_filter1[i][j]) == 0 and np.sum(
                            horiz_filter1[i][j]) == 0 and np.sum(vert_filter1[i][j]) == 0:
                        if "coast" in image_collection.image_list[r]:
                            sumB.append(image[i][j][2])
                        if "forest" in image_collection.image_list[r]:
                            sumB.append(image[i][j][2])
                        if "street" in image_collection.image_list[r]:
                            sumB.append(image[i][j][2])
            vert_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k1)
            horiz_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k2)
            back_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k3)
            forward_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k4)
            if "coast" in image_collection.image_list[r]:
                PixelCountCoastHigh.append(pixelCountHigh)
            if "forest" in image_collection.image_list[r]:
                PixelCountForestHigh.append(pixelCountHigh)
            if "street" in image_collection.image_list[r]:
                PixelCountStreetHigh.append(pixelCountHigh)
            mean = np.mean(sumB)
            if "coast" in image_collection.image_list[r]:
                sumBCoast.append(mean)
            if "forest" in image_collection.image_list[r]:
                sumBForest.append(mean)
            if "street" in image_collection.image_list[r]:
                sumBStreet.append(mean)

            count = count + 1
            print(count)
        # Create a plot
        thirdDimCoast, thirdDimForest, thirdDimStreet = image_collection.histAvgRegionGenHSV()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first set of data (x1, y1) with a blue line
        ax.scatter(PixelCountCoastHigh, sumBCoast, thirdDimCoast, label='Coast', color='blue')

        # Plot the second set of data (x2, y2) with a red line
        ax.scatter(PixelCountForestHigh, sumBForest, thirdDimForest, label='Forest', color='red')

        # Plot the third set of data (x3, y3) with a green line
        ax.scatter(PixelCountStreetHigh, sumBStreet, thirdDimStreet, label='Street', color='green')

        dataCorrelated = np.array(np.zeros((980,3)))
        for i in range(0,len(PixelCountCoastHigh)):
            dataCorrelated[i][0] = PixelCountCoastHigh[i]
            dataCorrelated[i][1] = sumBCoast[i]
            dataCorrelated[i][2] = thirdDimCoast[i]
        for i in range(len(PixelCountCoastHigh),len(PixelCountCoastHigh) + len(PixelCountForestHigh)):
            dataCorrelated[i][0] = PixelCountForestHigh[i-len(PixelCountCoastHigh)]
            dataCorrelated[i][1] = sumBForest[i-len(PixelCountCoastHigh)]
            dataCorrelated[i][2] = thirdDimForest[i-len(PixelCountCoastHigh)]
        for i in range(len(PixelCountCoastHigh) + len(PixelCountForestHigh),len(PixelCountCoastHigh) + len(PixelCountForestHigh) + len(PixelCountStreetHigh)):
            dataCorrelated[i][0] = PixelCountStreetHigh[i-(len(PixelCountCoastHigh)+len(PixelCountCoastHigh))]
            dataCorrelated[i][1] = sumBStreet[i-(len(PixelCountCoastHigh)+len(PixelCountCoastHigh))]
            dataCorrelated[i][2] = thirdDimStreet[i-(len(PixelCountCoastHigh)+len(PixelCountCoastHigh))]

        mat_cov = np.cov(dataCorrelated, rowvar= False)
        eig_vals, eig_vectors = np.linalg.eig(mat_cov)

        dataUnCorrelated = dataCorrelated @ eig_vectors
        # Add labels and legend

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # Show the plot
        plt.show()

        file_name = 'data/data_3classes_app/C1.txt'
        # Ouvrir le fichier en mode écriture
        with open(file_name, 'w') as file:
            # Parcourir les trois tableaux simultanément
            for a, b, c in zip(dataUnCorrelated[0:len(PixelCountCoastHigh),0], dataUnCorrelated[0:len(PixelCountCoastHigh),1], dataUnCorrelated[0:len(PixelCountCoastHigh),2]):
                # Écrire les éléments avec une séparation de trois espaces entre eux
                file.write(f"{a}   {b}   {c}\n")

        file_name = 'data/data_3classes_app/C2.txt'

        with open(file_name, 'w') as file:
            # Parcourir les trois tableaux simultanément
            for a, b, c in zip(dataUnCorrelated[len(PixelCountCoastHigh):len(PixelCountCoastHigh) + len(PixelCountForestHigh),0], dataUnCorrelated[len(PixelCountCoastHigh):len(PixelCountCoastHigh) + len(PixelCountForestHigh),1], dataUnCorrelated[len(PixelCountCoastHigh):len(PixelCountCoastHigh) + len(PixelCountForestHigh),2]):
                # Écrire les éléments avec une séparation de trois espaces entre eux
                file.write(f"{a}   {b}   {c}\n")

        file_name = 'data/data_3classes_app/C3.txt'
        # Ouvrir le fichier en mode écriture
        with open(file_name, 'w') as file:
            # Parcourir les trois tableaux simultanément
            for a, b, c in zip(dataUnCorrelated[len(PixelCountCoastHigh) + len(PixelCountForestHigh):len(PixelCountCoastHigh) + len(PixelCountForestHigh) + len(PixelCountStreetHigh),0], dataUnCorrelated[len(PixelCountCoastHigh) + len(PixelCountForestHigh):len(PixelCountCoastHigh) + len(PixelCountForestHigh) + len(PixelCountStreetHigh),1], dataUnCorrelated[len(PixelCountCoastHigh) + len(PixelCountForestHigh):len(PixelCountCoastHigh) + len(PixelCountForestHigh) + len(PixelCountStreetHigh),2]):
                # Écrire les éléments avec une séparation de trois espaces entre eux
                file.write(f"{a}   {b}   {c}\n")
    # Testing some classifiers
    if False:
        data3classes = ClassificationData()

        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        ppv1km1 = classifiers.PPVClassify_APP2(data2train=data3classes, data2test=data3classes,
                                               n_neighbors=1,
                                               experiment_title='1-PPV sur le 1-moy',
                                               useKmean=True, n_representants=7,
                                               gen_output=True, view=True)
        apriori = [1 / 3, 1 / 3, 1 / 3]
        cost = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
        bg1 = classifiers.BayesClassify_APP2(data2train=data3classes, data2test=data3classes,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True)

        n_neurons = 20
        n_layers = 6

        nn1 = classifiers.NNClassify_APP2(data2train=data3classes, data2test=data3classes,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='sigmoid',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.001),
                                          loss='categorical_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[
                                              K.callbacks.EarlyStopping(patience=50, verbose=1, restore_best_weights=1),
                                              classifiers.print_every_N_epochs(25)],
                                          # TODO à compléter L2.E4
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='3classes',
                                          ndonnees_random=5000, gen_output=True, view=True)
        plt.show()