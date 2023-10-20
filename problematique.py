"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt

from helpers.ImageCollection import ImageCollection


#######################################
def problematique_APP2():
    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    if True:
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        N = 6
        im_list = images.get_samples(N)
        #num 2.ici les images RGB ne sont pas vraiment stocker dans la memoire du programme elle sont plus-tôt référencer
        #dans un label list qu'on peut aller chercher la référence et l'afficher à l'aide de images_display
        #La représentation intermédiaire qui dois se faire dans images_display est une matrice de dimensions nXm des pixels
        #ou chaque pixel est représenter par 3 valeur RGB
        print(im_list)
        #images.images_display(im_list)
        #images.view_histogrammes(im_list)
        #images.generateRGBHistograms()
        #images.colorAvgAnalysis()
        #disc1 = images.histAvgRegionAnalysisLAB()
        disc1 = images.histAvgRegionGenLAB()
        #disc2 = images.histAvgRegionAnalysisRGB()
        disc2 = images.histAvgRegionGenRGB()
        #disc3 = images.histAvgRegionGenHSV()
        representationOfImages = []
        for i in range(0,len(images.image_list)):
            representationOfImages.append([disc2[i][2],disc1[i][1]],)

        fig = plt.figure()
        ax = fig.add_subplot()
        sizes = [50] * len(representationOfImages)
        colors = ['red' if i < 328 else 'blue' if 328 <= i < 688 else 'green' for i in
                  range(len(representationOfImages))]

        # Extract coordinates for each axis
        x_coords = [point[0] for point in representationOfImages]
        y_coords = [point[1] for point in representationOfImages]

        # Create a 3D scatter plot with specified point sizes and colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all points with their corresponding size and color
        ax.scatter(x_coords, y_coords, c=colors, s=sizes)

        # Set labels for each axis
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        # Display the plot
        plt.show()

    # TODO L1.E4.6 à L1.E4.8
    #images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
