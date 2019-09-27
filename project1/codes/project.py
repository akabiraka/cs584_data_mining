#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import math

#custom
from visualization import Visualization
from data_cleaning import DataCleaning

def main():
    print("This is main, alhumdulliah")
    ##### This block is for data cleaning #####
    missing_values = ["n/a", "na", "--", "?"]
    raw_data = pd.read_csv('../dataset_diabetes/diabetic_data.csv', delimiter=',', na_values = missing_values)
    #print(raw_data.head()) # print head of the data
    #print(raw_data.describe()) # shows numerical columns statistics e.g. count, mean, std, min, max etc
    #print(raw_data.shape) # prints shape of the dataset (101766, 50)
    #print(raw_data["weight"].isnull().sum()) #prints number of null values in weight column
    #print(raw_data["weight"].shape[0]) #prints number of columns in weight column
    data_cleaning = DataCleaning()
    raw_data = data_cleaning.clean_columns(raw_data, missing_bound=.2)
    cols_having_missing_values = data_cleaning.get_cols_having_missing_values(raw_data, False) # cols having missing values
    #raw_data.dtypes #shows the column data types
    raw_data = data_cleaning.fill_missing_values(raw_data, cols_having_missing_values)
    #print(get_cols_having_missing_values(raw_data, False)) #no columns with missing values
    print("Filled the missing values either by the mode or mean value")

    my_visu = Visualization()
    #my_visu.make_histograms(raw_data, ["race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id", "max_glu_serum", "A1Cresult", "num_lab_procedures", "time_in_hospital"], image_path="../outputs/attr_hist_plot.png") # all the attributes distributions
    #my_visu.make_histograms(raw_data, ["change", "diabetesMed", "readmitted"], figsize=(15, 15), cmap="spring", image_path="../outputs/class_attr_hist_plot.png") # class attribute-value distributions
    my_visu.build_scatter_2attr_plot(raw_data, x_col="num_procedures", y_col="num_medications", image_path="../outputs/num_of_procedures_vs_medications_plot.png")


if __name__ == "__main__":
    main()
