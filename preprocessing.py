import cv2
import numpy as np

IMG_SIZE = 64

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    return image

def extract_features(image_path):
    image = preprocess_image(image_path)
    return image.flatten()