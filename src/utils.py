import matplotlib.pyplot as plt
import numpy as np

def plot_results(model_history):
    train_accuracy = model_history.history['accuracy']
    train_loss = model_history.history['loss']
    validation_accuracy = model_history.history['val_accuracy']
    validation_loss = model_history.history['val_loss']
    plt.plot(train_loss); plt.plot(validation_loss)
    plt.xlabel('Number of Epochs'); plt.ylabel('Loss'); plt.legend(['Train', 'Validation'])
    plt.figure()
    plt.plot(train_accuracy); plt.plot(validation_accuracy)
    plt.xlabel('Number of Epochs'); plt.ylabel('Accuracy'); plt.legend(['Train', 'Validation'])



def plot_predicts(model, X_test, test_labels, test_images, number_of_predictions):
    test_labels_pred = model.predict(X_test)
    test_labels_pred = np.argmax(test_labels_pred, axis=1)
    n = 0
    _, axs = plt.subplots(1, number_of_predictions, figsize=(30, 30))
    for i in range(len(test_labels)):
        if n >= number_of_predictions:
            break
        if (test_labels_pred[i] == test_labels[i]):
            axs[n].imshow(test_images[i], cmap='gray')
            axs[n].set_title(f'{test_labels[i]} -> {test_labels_pred[i]}')
            axs[n].axis('off')
            n += 1