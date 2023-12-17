import numpy as np

def create_array(size):
    array = np.array([])
    for i in range(size):
        array = np.append(array, 0)
    return array

def array_from_out_array(out_array, iterations):
    array = create_array(16)
    for i in range(16):
        array[i] = out_array[i + iterations]
    return array

def counting_iterations(out_array):
    n_bits = len(out_array)
    iterations = n_bits - 16
    return iterations

# model neuronet
class Model:
    def __init__(self):
        self._hid_neuro = 16  # one hidden layer of neurons
        self._previous_pred = create_array(16)
        self._out_array = []
        self._array = []
        self._weight0 = []
        self._weight1 = []
        self._iterations = 0
        self._pred = create_array(16)

    def add_array(self):
        pass

    def predict(self):
        pass

class ModelCod(Model):
    def __init__(self):
        self.__input_neuro = 16
        self.__output_neuro = 8192

    def add_array(self, new_array, weight_to_16, weight_to_8192):
        self._out_array = new_array
        self._weight0 = weight_to_16
        self._weight1 = weight_to_8192
        self._iterations = counting_iterations(self._out_array)

    def predict(self):
        for i in range(self._iterations):
            self._array = array_from_out_array(self._out_array, i)
            self._pred = np.dot(self._array, self._weight0)
            self._pred = np.tanh(self._pred)
            self._pred += self._previous_pred
            self._previous_pred = self._pred
        self._pred = np.dot(self._pred, self._weight1)
        return self._pred

class ModelDecod(Model):
    def __init__(self):
        self.__input_neuro = 8192
        self.__output_neuro = 16

    def add_array(self, new_array, weight_to_8192, weiqht_to_16, iterations):
        self._out_array = new_array
        self._weight0 = weight_to_8192
        self._weight1 = weiqht_to_16
        self._iterations = iterations

    def predict(self):
        self._pred = np.dot(self._out_array, self._weight0)
        for i in range(self._iterations + 1):
            self._pred = np.tanh(self._pred)
            self._pred += self._previous_pred
            self._previous_pred = self._pred
            self._pred = np.dot(self._pred, self._weight1)
            return self._pred

# view data for learning and etc
class View:
    pass

# controller of neuronet
class Controller:
    def __init__(self):
        self.code = ModelCod()
        self.decode = ModelDecod()