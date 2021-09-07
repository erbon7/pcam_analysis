
import matplotlib.pyplot as plt
import numpy as np

def plot_figures(fpr, tpr, history, auc, roc_fn, loss_fn, accuracy_fn): 

    lw = 2
    dpi = 150
    plt.figure()
    plt.plot(fpr, tpr, lw=lw, label="ROC curve (area = {:0.2f})".format(auc), color='darkorange') 
    plt.plot([0, 1], [0, 1], label='random guessing', lw=lw, linestyle='--', color='navy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(roc_fn, dpi=dpi)

    plt.figure()
    plt.plot(history.history['loss'], lw=lw)
    plt.plot(history.history['val_loss'], lw=lw)
    plt.title('model loss')
    plt.ylim([0.0, 1.05])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(loss_fn, dpi=dpi)

    plt.figure()
    plt.plot(history.history['acc'], lw=lw)
    plt.plot(history.history['val_acc'], lw=lw)
    plt.ylim([0.0, 1.05])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(accuracy_fn, dpi=dpi)


def load_norm_data():

    train_data = np.load("pcam_train_data.npz")
    valid_data = np.load("pcam_valid_data.npz")
    test_data = np.load("pcam_test_data.npz")

    # normalization
    x_train = train_data['x_train'].astype('float32')
    x_train /= 255

    x_valid = valid_data['x_valid'].astype('float32')
    x_valid /= 255

    x_test = test_data['x_test'].astype('float32')
    x_test /= 255

    y_train = train_data['y_train']
    y_valid = valid_data['y_valid']
    y_test = test_data['y_test']

    return(x_train, x_valid, x_test, y_train, y_valid, y_test)



