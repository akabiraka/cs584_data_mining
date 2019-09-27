
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self, df=None):
        self.df = df

    def hello(self):
        return "Alhumdulillah, hello"

    def save_and_show_plot(self, path=None, is_show=True):
        if path:
            plt.savefig(path)
        if is_show:
            plt.show()

    def make_histograms(self, df, columns=None, figsize=(15,15), cmap=None, image_path=None, is_show=True):
        """
        This function make histograms for each column name. Each of the column should be 'Nominal'
        Parameters
        ----------
        df: pandas dataframe
            Pandas dataframe associated with column names.
        columns: dataframe column names in array
            e.g. ['aColumn', 'bColumn']
        figsize: default=(15, 15)
        cmap: colormap codes for for custom colored bar or histogram
            defautls are:
                summer or green for numerical
                winter or blue for categorical
        Effects
        -------
        Creates columns.size() number of histograms.
        """
        nrows = 3
        ncols = 3
        plt.figure(figsize=figsize)
        for i, col in enumerate(df[columns]):
            plt.subplot(nrows,ncols,i+1)
            if is_numeric_dtype(df[col]):
                df[col].plot(kind="hist", colormap=(cmap or "summer")).set_title(col)
            else:
                df[col].value_counts().plot(kind="bar", colormap=(cmap or "winter")).set_title(col)

        plt.subplots_adjust(top = .95, bottom=.05, hspace=.5, wspace=0.4)

        self.save_and_show_plot(image_path, is_show)

    def build_scatter_2attr_plot(self, df, x_col, y_col, figsize=(20, 20), cmap=None, image_path=None, is_show=True):
        #for i, col in enumerate(columns):
        ax = df.plot(x=x_col, y=y_col, style='o')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        self.save_and_show_plot(image_path, is_show)
