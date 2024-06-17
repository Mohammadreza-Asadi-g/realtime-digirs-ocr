from tensorflow.keras.models import load_model
from src.utils import read_config


config = read_config()
model = load_model(config["training_data"]["model_save_path"] + 'model.h5')

def ocr(digits_concat):
    segmented_digits = digits_concat.reshape(len(digits_concat), 28, 28, 1)
    segmented_digits = segmented_digits.astype('float32') / 255
    labels = model.predict(segmented_digits)
    return labels