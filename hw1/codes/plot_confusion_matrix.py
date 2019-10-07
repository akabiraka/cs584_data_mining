#!/usr/bin/env python
# coding: utf-8

# In[11]:


print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#get_ipython().run_line_magic('matplotlib', 'inline')
def plotConfusionMatrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None, image_path=None,
                          figsize=(20,20),
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    #ax.set_ylim(10.0, 0)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    if image_path:
        plt.savefig(image_path)
    #plt.show()
    return ax


#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      #title='Normalized confusion matrix')

#plt.show()




# In[4]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_my_conf_matrix(y_true, y_pred, classes, figsize=(10, 10), save_path=None, title="Confusion matrix", cmap=plt.cm.Reds):
    """This calculates confusion matrix and plot that."""

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize=figsize)
    sn.set(font_scale=1.4)
    ax = sn.heatmap(df_cm, annot=True, fmt="g", cmap=cmap)
    ax.set_ylim(10, 0)
    ax.set(xlabel='Predicted label', ylabel='True label', title=title)
    if save_path:
        plt.savefig(save_path)


# In[13]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def my_test():
    array = [[13,1,1,0,2,0],
     [3,9,6,0,1,0],
     [0,0,16,2,0,0],
     [0,0,0,13,0,0],
     [0,0,0,0,15,0],
     [0,0,1,0,0,15]]
    df_cm = pd.DataFrame(array, range(6), range(6))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    #plt.savefig("xxx.png")
    ax.set_ylim(6.0, 0)
    #plt.show()

my_test()


# In[ ]:
