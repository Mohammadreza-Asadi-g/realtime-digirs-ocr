from tensorflow.keras import models, layers

def cnn_model():
    input = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(input)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
    flatten = layers.Flatten()(conv2)
    output = layers.Dense(10, activation='softmax')(flatten)
    model = models.Model(input, output)
    return model