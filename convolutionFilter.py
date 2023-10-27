import numpy as np
from helpers.ImageCollection import ImageCollection
import cv2

def normalize_hist_rgb(image):
    im = np.swapaxes(image, 0,2)
    equalized = []
    for channel in im:
        equalized.append(cv2.equalizeHist(channel))
    return np.swapaxes(np.array(equalized), 0, 2)

def get_centroid(image):
    running_sum = []
    total_v = 0
    gray_scale = np.sum(cv2.transpose(image), axis=2) + 1
    log_scale = np.emath.logn(np.max(image), gray_scale)
    shape = log_scale.shape
    threshold = 0.9
    for (x,y), v in np.ndenumerate(log_scale):
        if v > threshold:
            total_v += v
            running_sum.append(v * np.array([x, y]))
    running_sum = np.swapaxes(running_sum, 0, 1) / total_v
    cov = np.cov(running_sum)
    eig_vals, eig_vects = np.linalg.eig(cov)
    return np.sum(running_sum, axis=1)

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
        vert_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k1)
        horiz_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k2)
        back_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k3)
        forward_diag_filter = cv2.filter2D(src=image, ddepth=-1, kernel=k4)
        image_filtered = cv2.bilateralFilter(image,9,75,75)
        centroid = get_centroid(vert_filter)
        vert_filter = cv2.drawMarker(vert_filter, get_centroid(vert_filter).astype('uint32'), color=(200,0,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        horiz_filter = cv2.drawMarker(horiz_filter, get_centroid(horiz_filter).astype('uint32'), color=(200,0,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        back_diag_filter = cv2.drawMarker(back_diag_filter, get_centroid(back_diag_filter).astype('uint32'), color=(200,0,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        forward_diag_filter = cv2.drawMarker(forward_diag_filter, get_centroid(forward_diag_filter).astype('uint32'), color=(200,0,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        res = np.hstack((image, vert_filter, horiz_filter, back_diag_filter, forward_diag_filter))
        cv2.imshow("original", res)
        cv2.waitKey()
        cv2.destroyAllWindows()