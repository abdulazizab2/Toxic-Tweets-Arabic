import numpy as np
import matplotlib.pyplot as plt
import itertools

def tensorflow_to_numpy(ds, model):
  """
    Iterate over a tensorflow dataset object to get the predicted class and the true class as numpy objects
    @params
    ds: dataset object
    @return
    true_classes: true classses
    predicted_classes: predicted classes
  """
  predicted_classes = np.array([])
  true_classes =  np.array([])

  for x, y in ds:
    predicted_classes = np.concatenate([predicted_classes,
                        np.argmax(model(x), axis = -1)])
    true_classes = np.concatenate([true_classes, np.argmax(y.numpy(), axis=-1)])

  return true_classes, predicted_classes

def plotComplexity(history, plot_accuracy=False):

    if plot_accuracy == True:

        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        train_accuracy = history.history['categorical_accuracy']
        val_accuracy = history.history['val_categorical_accuracy']
        fig, axes = plt.subplots(2, 1, figsize=(14,10))
        axes[0].plot(train_loss, label="Train Loss")
        axes[0].plot(val_loss, label="Validation Loss")
        axes[0].legend()
        axes[0].set_title("Train Loss vs Validation Loss")
        axes[1].plot(train_accuracy, label='Train Accuracy')
        axes[1].plot(val_accuracy, label='Validation Accuracy')
        axes[1].legend()
        axes[1].set_title('Train Accuracy vs Validation Accuracy')
        plt.show()

    elif plot_accuracy == False:

        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Val Loss')
        plt.legend()
        plt.title('Train Loss vs Validation Loss')
        plt.show()

def plot_cm(cm, class_names):
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure