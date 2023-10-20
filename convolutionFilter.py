import numpy as np
from helpers.ImageCollection import ImageCollection
import cv2

def normalize_hist_rgb(image):
    im = np.swapaxes(image, 0,2)
    equalized = []
    for channel in im:
        equalized.append(cv2.equalizeHist(channel))
    return np.swapaxes(np.array(equalized), 0, 2)

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
                    [edge_coef, 1,  edge_coef],
                    [edge_coef, edge_coef, 1]])
    k4 = np.array([[edge_coef, edge_coef, 1],
                    [edge_coef, 1, edge_coef],
                    [1, edge_coef, edge_coef]])
    
    # filter2D() function can be used to apply kernel to an image.
    # Where ddepth is the desired depth of final image. ddepth is -1 if...
    # ... depth is same as original or source image.
    N_samples = 5
    images_indexes = image_collection.get_samples(N_samples)
    images = image_collection.load_images(images_indexes)
    for i, image in enumerate(images):
        image_normalized = normalize_hist_rgb(image)
        vert_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k1)
        horiz_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k2)
        back_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k3)
        forward_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k4)
        image_filtered = cv2.bilateralFilter(image,9,75,75)
        res = np.hstack((image, image_filtered, horiz_filter, back_diag_filter, forward_diag_filter))
        cv2.imshow("original", res)
        cv2.waitKey()
        cv2.destroyAllWindows()