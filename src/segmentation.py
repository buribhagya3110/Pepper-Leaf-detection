import cv2
import numpy as np

def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented