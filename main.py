import numpy as np

# model neuronet
class Model:
    def __init__(self):
        self._hid_neuro = 16  # one hidden layer of neurons
        self._previous_pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self._array = []
        self._weight0 = []
        self._weight1 = []
        self._iterations = 0
        self._pred = self._previous_pred

    def add_values(self):
        pass

    def counting_iterations(self):
        pass

    def predict(self):
        pass

class ModelCod(Model):
    def __init__(self):
        self.__input_neuro = 16
        self.__output_neuro = 8192

    def add_values(self, new_array, weight_to_16, weihgt_to_8192):
        self._array.append(new_array)
        self._weight0.append(weight_to_16)
        self._weight1.append(weihgt_to_8192)

    def counting_iterations(self):
        n_bits = len(self._array)
        self._iterations = n_bits / 16
        return self._iterations

    def predict(self):
        for i in range(self._iterations):
            self._pred = np.dot(self._array, self._weight_to_16)
            self._pred = self._pred + self._previous_pred
            self._pred = np.tanh(self._pred)
            self._previous_pred = self._pred
        self._pred = np.dot(self._pred, self._weihgt_to_8192)
        return self._pred

class ModelDecod(Model):
    def __init__(self):
        self.__input_neuro = 8192
        self.__output_neuro = 16

    def add_values(self, new_array, weight_to_8192, weiqht_to_16, iterations):
        self._array.append(new_array)
        self._weight0.append(weight_to_8192)
        self._weight1.append(weiqht_to_16)
        self._iterations = iterations

    def predict(self):
        self._pred = np.dot(self._array, self._weight0)
        for i in range(self._iterations):
            self._pred = self._pred + self._previous_pred
            self._pred = np.tanh(self._pred)
            self._previous_pred = self._pred
            self._pred = np.dot(self._pred, self._weight1)
            return self._pred

# view data for learning and etc
class View:
    pass

# controller of neuronet
class Controller:
    pass