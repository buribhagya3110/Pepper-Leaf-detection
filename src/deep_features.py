import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_deep_features(image):
    img = np.expand_dims(image, axis=0)
    img = preprocess_input(img)
    
    features = base_model.predict(img)
    
    return features.flatten()