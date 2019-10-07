#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn import tree
import sklearn.metrics as metrics # accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

import import_ipynb
from plot_confusion_matrix import plot_my_conf_matrix as conf_x
from util import Util
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:

def load_data():
    # loading train and test data and cleaning missing values
    raw_df_train = pd.read_csv('../pendigits_data/pendigits.tra', delimiter=',', header=None)
    raw_df_test = pd.read_csv('../pendigits_data/pendigits.tes', delimiter=',', header=None)
    # print(raw_df_train.head()) # prints sample of the dataset
    train_df_clean = raw_df_train.dropna() # drop any rows with missing values
    print("Dropped rows in train set: %d" %(raw_df_train.shape[0] - train_df_clean.shape[0])) # number of rows dropped for some missing values
    # print(train_df_clean.describe()) # prints statistics column wise for train data

    test_df_clean = raw_df_test.dropna()
    print("Dropped rows in train set: %d" %(raw_df_test.shape[0] - test_df_clean.shape[0])) # number of rows dropped for some missing values
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
    print("Accuracy: ", metrics.accuracy_score(y_true, y_predict)) # accuracy score
    print("Precition per class: ", metrics.precision_score(y_true, y_predict, average=None)) # precision scores for each class
    print("Precision: ", metrics.precision_score(y_true, y_predict, average='macro')) # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account,
    print("Recall per class: ", metrics.recall_score(y_true, y_predict, average=None)) # recall score for each class
    print("Recall: ", metrics.recall_score(y_true, y_predict, average='macro'))

def get_dt_model(x_train, y_train, criterion="gini"):
    model = tree.DecisionTreeClassifier(criterion=criterion)
    model.fit(x_train, y_train)
    return model

def predict(model, x_data, y_data, save_path=None):
    y_predict = model.predict(x_data)
    print_accuray_precision_recall(y_data, y_predict)
    if save_path:
        conf_x(y_data, y_predict, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], title="Confusion Matrix for Train set (gini)", save_path=save_path)

def k_fold_cross_validation(x_train, y_train, k=5, criterion='gini'):
    model = tree.DecisionTreeClassifier(criterion=criterion)
    print(cross_val_score(model, x_train, y_train, cv=k))

def main():
    x_train, y_train, x_test, y_test = load_data()
    gini_model = get_dt_model(x_train, y_train, criterion='gini')
    predict(gini_model, x_train, y_train, save_path="../graphs/conf_matrix_gini_train_set.png")
    predict(gini_model, x_test, y_test, save_path="../graphs/conf_matrix_gini_test_set.png")
    entropy_model = get_dt_model(x_train, y_train, criterion='entropy')
    predict(gini_model, x_train, y_train, save_path="../graphs/conf_matrix_entropy_train_set.png")
    predict(gini_model, x_test, y_test, save_path="../graphs/conf_matrix_entropy_test_set.png")
    k_fold_cross_validation(x_train, y_train, k=5, criterion='gini')
    k_fold_cross_validation(x_train, y_train, k=5, criterion='entropy')
    util = Util()
    util.save_model(model=gini_model, filename="../saved_models/gini_model.sav")
    util.save_model(model=entropy_model, filename="../saved_models/entropy_model.sav")
main()
