#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn import tree
# accuracy_score, precision_score, recall_score, confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

import import_ipynb
from plot_confusion_matrix import plot_my_conf_matrix as conf_x
from util import Util
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:

def load_data():
    # loading train and test data and cleaning missing values
    raw_df_train = pd.read_csv(
        '../pendigits_data/pendigits.tra', delimiter=',', header=None)
    raw_df_test = pd.read_csv(
        '../pendigits_data/pendigits.tes', delimiter=',', header=None)
    # print(raw_df_train.head()) # prints sample of the dataset
    train_df_clean = raw_df_train.dropna()  # drop any rows with missing values
    # number of rows dropped for some missing values
    print("Dropped rows in train set: %d" %
          (raw_df_train.shape[0] - train_df_clean.shape[0]))
    # print(train_df_clean.describe()) # prints statistics column wise for train data

    test_df_clean = raw_df_test.dropna()
    # number of rows dropped for some missing values
    print("Dropped rows in train set: %d" %
          (raw_df_test.shape[0] - test_df_clean.shape[0]))
    # print(test_df_clean.describe()) # prints statistics column wise for test data
    # seperating the label column for train and test set
    x_train = train_df_clean.drop(axis=1, columns=[16])
    y_train = train_df_clean.iloc[:, 16]

    x_test = test_df_clean.drop(axis=1, columns=[16])
    y_test = test_df_clean.iloc[:, 16]

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def print_accuray_precision_recall(y_true, y_predict):
    """Prints accuray, presion for each classes and recall for each classes."""
    print("Accuracy: ", metrics.accuracy_score(
        y_true, y_predict))  # accuracy score
    print("Precition per class: ", metrics.precision_score(
        y_true, y_predict, average=None))  # precision scores for each class
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account,
    print("Precision: ", metrics.precision_score(
        y_true, y_predict, average='macro'))
    print("Recall per class: ", metrics.recall_score(
        y_true, y_predict, average=None))  # recall score for each class
    print("Recall: ", metrics.recall_score(y_true, y_predict, average='macro'))


def get_dt_model(x_train, y_train, criterion='gini', splitter='best', max_depth=None, max_leaf_nodes=None):
    model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter,
                                        max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(x_train, y_train)
    print("Tree: ", model.tree_)
    return model


def predict(model, x_data, y_data, save_path=None, title=None):
    y_predict = model.predict(x_data)
    print_accuray_precision_recall(y_data, y_predict)
    if save_path:
        conf_x(y_data, y_predict, classes=[
               '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], title=title, save_path=save_path)


def k_fold_cross_validation(x_train, y_train, k=5, criterion='gini'):
    model = tree.DecisionTreeClassifier(criterion=criterion)
    print(cross_val_score(model, x_train, y_train, cv=k))


def main():
    x_train, y_train, x_test, y_test = load_data()
    util = Util()

    print("DT: criterion='gini', splitter='best', max_depth=None, max_leaf_nodes=None")
    gini_model = get_dt_model(
        x_train, y_train, criterion='gini', splitter='best')
    util.save_model(model=gini_model,
                    filename="../saved_models/gini_bestsplitter_model.sav")
    print("Train set")
    predict(gini_model, x_train, y_train, save_path="../graphs/gini_bestsplitter_train_set.png",
            title="Confusion matrix (gini, best-splitter, train-set)")
    print("Test set")
    predict(gini_model, x_test, y_test, save_path="../graphs/gini_bestsplitter_test_set.png",
            title="Confusion matrix (gini, best-splitter, test-set)")

    print("DT: criterion='gini', splitter='random', max_depth=None, max_leaf_nodes=None")
    gini_model = get_dt_model(
        x_train, y_train, criterion='gini', splitter='random')
    util.save_model(model=gini_model,
                    filename="../saved_models/gini_randomsplitter_model.sav")
    print("Train set")
    predict(gini_model, x_train, y_train, save_path="../graphs/gini_randomsplitter_train_set.png",
            title="Confusion matrix (gini, random-splitter, train-set)")
    print("Test set")
    predict(gini_model, x_test, y_test, save_path="../graphs/gini_randomsplitter_test_set.png",
            title="Confusion matrix (gini, random-splitter, test-set)")

    print("DT: criterion='entropy', splitter='best', max_depth=None, max_leaf_nodes=None")
    entropy_model = get_dt_model(
        x_train, y_train, criterion='entropy', splitter='best')
    util.save_model(model=entropy_model,
                    filename="../saved_models/entropy_bestsplitter_model.sav")
    print("Train set")
    predict(entropy_model, x_train, y_train, save_path="../graphs/entropy_bestsplitter_train_set.png",
            title="Confusion matrix (entropy, best-splitter, train-set)")
    print("Test set")
    predict(entropy_model, x_test, y_test, save_path="../graphs/entropy_bestsplitter_test_set.png",
            title="Confusion matrix (entropy, best-splitter, test-set)")

    print("DT: criterion='entropy', splitter='random', max_depth=None, max_leaf_nodes=None")
    entropy_model = get_dt_model(
        x_train, y_train, criterion='entropy', splitter='random')
    util.save_model(model=entropy_model,
                    filename="../saved_models/entropy_randomsplitter_model.sav")
    print("Train set")
    predict(entropy_model, x_train, y_train, save_path="../graphs/entropy_randomsplitter_train_set.png",
            title="Confusion matrix (entropy, random-splitter, train-set)")
    print("Test set")
    predict(entropy_model, x_test, y_test, save_path="../graphs/entropy_randomsplitter_test_set.png",
            title="Confusion matrix (entropy, random-splitter, test-set)")

    print("DT: criterion='gini', splitter='best', max_depth=10, max_leaf_nodes=None")
    gini_model = get_dt_model(
        x_train, y_train, criterion='gini', splitter='best', max_depth=10)
    util.save_model(model=gini_model,
                    filename="../saved_models/gini_bestsplitter_maxDepth10_model.sav")
    print("Train set")
    predict(gini_model, x_train, y_train, save_path="../graphs/gini_bestsplitter_maxDepth10_train_set.png",
            title="Confusion matrix (gini, best-splitter, maxDepth=10, train-set)")
    print("Test set")
    predict(gini_model, x_test, y_test, save_path="../graphs/gini_bestsplitter_maxDepth10_test_set.png",
            title="Confusion matrix (gini, best-splitter, maxDepth=10, test-set)")

    print("DT: criterion='gini', splitter='random', max_depth=10, max_leaf_nodes=None")
    gini_model = get_dt_model(
        x_train, y_train, criterion='gini', splitter='random', max_depth=10)
    util.save_model(model=gini_model,
                    filename="../saved_models/gini_randomsplitter_maxDepth10_model.sav")
    print("Train set")
    predict(gini_model, x_train, y_train, save_path="../graphs/gini_randomsplitter_maxDepth10_train_set.png",
            title="Confusion matrix (gini, random-splitter, maxDepth=10, train-set)")
    print("Test set")
    predict(gini_model, x_test, y_test, save_path="../graphs/gini_randomsplitter_maxDepth10_test_set.png",
            title="Confusion matrix (gini, random-splitter, maxDepth=10, test-set)")

    print("DT: criterion='entropy', splitter='best', max_depth=10, max_leaf_nodes=None")
    entropy_model = get_dt_model(
        x_train, y_train, criterion='entropy', splitter='best', max_depth=10)
    util.save_model(model=entropy_model,
                    filename="../saved_models/entropy_bestsplitter_maxDepth10_model.sav")
    print("Train set")
    predict(entropy_model, x_train, y_train, save_path="../graphs/entropy_bestsplitter_maxDepth10_train_set.png",
            title="Confusion matrix (entropy, best-splitter, maxDepth=10, train-set)")
    print("Test set")
    predict(entropy_model, x_test, y_test, save_path="../graphs/entropy_bestsplitter_maxDepth10_test_set.png",
            title="Confusion matrix (entropy, best-splitter, maxDepth=10, test-set)")

    print("DT: criterion='entropy', splitter='random', max_depth=10, max_leaf_nodes=None")
    entropy_model = get_dt_model(
        x_train, y_train, criterion='entropy', splitter='random', max_depth=10)
    util.save_model(model=entropy_model,
                    filename="../saved_models/entropy_randomsplitter_maxDepth10_model.sav")
    print("Train set")
    predict(entropy_model, x_train, y_train, save_path="../graphs/entropy_randomsplitter_maxDepth10_train_set.png",
            title="Confusion matrix (entropy, random-splitter, maxDepth=10, train-set)")
    print("Test set")
    predict(entropy_model, x_test, y_test, save_path="../graphs/entropy_randomsplitter_maxDepth10_test_set.png",
            title="Confusion matrix (entropy, random-splitter, maxDepth=10, test-set)")

    k_fold_cross_validation(x_train, y_train, k=5, criterion='gini')
    k_fold_cross_validation(x_train, y_train, k=5, criterion='entropy')


main()
