
import pickle

class Util():
    """docstring for My_util."""

    def __init__(self, arg=None):
        self.arg = arg

    def save_model(self, model, filename):
        # filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self, filename):
        return pickle.load(open(filename, 'rb'))
