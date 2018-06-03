print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score

def get_cnf_mat(gt_test, pred):
    class_names = ['Low','Mid','High']

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(gt_test, pred)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix,classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    return cnf_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')# Compute confusion matrix

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_classes(labels):
    classes = ['Low','Mid','High']
    len0 = np.flatnonzero(labels == 0).shape[0]
    len1 = np.flatnonzero(labels == 1).shape[0]
    len2 = np.flatnonzero(labels == 2).shape[0]
    lens = [len0,len1,len2]

    y_pos = np.arange(len(classes))
    fig, ax = plt.subplots()
    ax.barh(y_pos, lens, align='center',
        color='blue')
    for i, v in enumerate(lens):
        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Sample size')
    ax.set_title('Class distribution over the PANDORA dataset')
    plt.show()