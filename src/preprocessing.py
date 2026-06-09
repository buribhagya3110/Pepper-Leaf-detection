import cv2

def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Image not found or corrupted: {path}")

    img = cv2.resize(img, (224, 224))
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img