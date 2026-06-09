import cv2
import numpy as np

def extract_color_features(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    features = []

    # Mean values
    for channel in cv2.split(lab):
        features.append(np.mean(channel))

    for channel in cv2.split(hsv):
        features.append(np.mean(channel))

    # Vegetation Index (Excess Green)
    B, G, R = cv2.split(image)
    exg = 2 * G - R - B
    features.append(np.mean(exg))

    return np.array(features)