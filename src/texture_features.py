import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # GLCM
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # LBP
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)

    return np.hstack([contrast, energy, homogeneity, lbp_hist])