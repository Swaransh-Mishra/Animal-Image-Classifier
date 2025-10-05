from PIL import Image
import numpy as np

def preprocess_image(img, target_size=(224, 224)):
    """
    Preprocess uploaded PIL image for model prediction.
    Resizes, scales pixel values and adds batch dimension.
    """
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array
