
import pickle
from plot_confusion_matrix import plot_my_conf_matrix as conf_x
import sklearn.metrics as metrics

class Util():
    """docstring for My_util."""

    def __init__(self, arg=None):
        self.arg = arg

    def save_model(self, model, filename):
        # filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self, filename):
        return pickle.load(open(filename, 'rb'))

    def print_accuray_precision_recall(self, y_true, y_predict):
        """Prints accuray, presion for each classes and recall for each classes."""
        print("Accuracy: ", metrics.accuracy_score(y_true, y_predict)) # accuracy score
        print("Precition per class: ", metrics.precision_score(y_true, y_predict, average=None)) # precision scores for each class
        print("Precision: ", metrics.precision_score(y_true, y_predict, average='macro')) # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account,
        print("Recall per class: ", metrics.recall_score(y_true, y_predict, average=None)) # recall score for each class
        print("Recall: ", metrics.recall_score(y_true, y_predict, average='macro'))

    def predict(self, model, x_data, y_data, save_path=None, title=None):
        y_predict = model.predict(x_data)
        print(y_predict)
        self.print_accuray_precision_recall(y_data, y_predict)
        if save_path:
            conf_x(y_data, y_predict, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], title=title, save_path=save_path)
