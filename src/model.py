from tensorflow.keras import models, layers

def cnn_model():
    input = layers.Input(shape=(28, 28, 1))
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(input)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
    flatten = layers.Flatten()(conv2)
    output = layers.Dense(10, activation='softmax')(flatten)
    model = models.Model(input, output)
    return model


def cnn_model_2():
    input = layers.Input(shape=(100, 100, 1))
    try:
        x = layers.Rescaling(1./255)(input)
    except:
        x = layers.experimental.preprocessing.Rescaling(1./255)(input)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inputs=input, outputs=output)    