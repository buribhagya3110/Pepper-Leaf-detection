import numpy as np

def fuse_features(color, texture, deep):
    return np.concatenate([color, texture, deep])