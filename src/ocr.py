from tensorflow.keras.models import load_model
from src.utils import read_config


config = read_config()
model = load_model(config["training_data"]["model_save_path"] + 'model.h5')

def ocr(image):
    X_test = image.reshape(len(image), 28, 28, 1)
    label = model.predict(X_test)
    return label