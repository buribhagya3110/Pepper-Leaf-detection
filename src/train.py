import os
import numpy as np

from preprocessing import preprocess_image
from color_features import extract_color_features
from texture_features import extract_texture_features
from deep_features import extract_deep_features
from fusion import fuse_features
from model import train_model
from scaler import fit_scaler

base_path = "data"

X, y = [], []

label_map = {
    "healthy": 0,
    "bacterial_spot": 1
}

for label in label_map.keys():

    color_dir = os.path.join(base_path, "color", label)
    gray_dir = os.path.join(base_path, "grayscale", label)
    seg_dir = os.path.join(base_path, "segmented", label)

    color_files = sorted(os.listdir(color_dir))
    gray_files = sorted(os.listdir(gray_dir))
    seg_files = sorted(os.listdir(seg_dir))

    # 🔥 ensure same length
    min_len = min(len(color_files), len(gray_files), len(seg_files))

    for i in range(min_len):
        try:
            color_path = os.path.join(color_dir, color_files[i])
            gray_path = os.path.join(gray_dir, gray_files[i])
            seg_path = os.path.join(seg_dir, seg_files[i])

            color_img = preprocess_image(color_path)
            gray_img = preprocess_image(gray_path)
            seg_img = preprocess_image(seg_path)

            color_feat = extract_color_features(color_img)
            texture_feat = extract_texture_features(gray_img)
            deep_feat = extract_deep_features(seg_img)

            features = fuse_features(color_feat, texture_feat, deep_feat)

            X.append(features)
            y.append(label_map[label])

        except Exception as e:
            print(f"Skipping {i}: {e}")

X = np.array(X)
y = np.array(y)

X = fit_scaler(X)

model = train_model(X, y)