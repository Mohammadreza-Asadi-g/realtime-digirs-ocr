from tensorflow.keras.models import load_model
import yaml

with open("./configs/configurations.yaml", 'r') as file:
    config = yaml.safe_load(file)

model = load_model(config["training_data"]["model_save_path"] + 'model.h5')

def ocr(image):
    X_test = image.reshape(len(image), 28, 28, 1)
    label = model.predict(X_test)
    return label