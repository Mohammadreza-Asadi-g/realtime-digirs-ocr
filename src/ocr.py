from tensorflow.keras.models import load_model
try:
    from src.utils import read_config
except:
    from utils import read_config
import numpy as np


config = read_config()
model = load_model(config["training_data"]["model_save_path"] + 'model.h5')

def ocr(digits_concat):
    segmented_digits = digits_concat.reshape(len(digits_concat), 28, 28, 1)
    segmented_digits = segmented_digits.astype('float32') / 255
    labels = model.predict(segmented_digits)
    labels = np.argmax(labels, axis=1)
    return labels

if __name__ == '__main__':
    from segmentation import digits_segmentaion
    import cv2 as cv
    image = cv.imread(config["inference"]["images_path"] + "numbers_1.jpg")
    digits_concat, contours_min_location = digits_segmentaion(image,
                                       config["inference"]["segmentation_threshold_value"],
                                       config["inference"]["segmentation_digit_min_area"],
                                       config["inference"]["segmentation_digit_crop_offset"])
    labels = ocr(digits_concat)
    print(labels)