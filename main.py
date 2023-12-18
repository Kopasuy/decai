import numpy as np

array_weight_2 = np.random.randn(2, 2)
array_weight_1024 = np.random.randn(2, 1024)

def create_array(size):
    array = np.array([])
    for i in range(size):
        array = np.append(array, 0)
    return array

def array_from_out_array(out_array, iterations):
    array = create_array(2)
    for i in range(2):
        array[i] = out_array[i + iterations]
    return array

def counting_iterations(out_array):
    n_byte = len(out_array)
    iterations = n_byte - 2
    return iterations

# model neuronet
class Model:
    def __init__(self):
        self._hid_neuro = 2  # one hidden layer of neurons
        self._previous_pred = create_array(2)
        self._out_array = []
        self._array = []
        self._weight0 = []
        self._weight1 = []
        self._iterations = 0
        self._pred = create_array(2)

    def add_array(self):
        pass

    def predict(self):
        pass

class ModelCod(Model):
    def __init__(self):
        self.__input_neuro = 2
        self.__output_neuro = 1024

    def add_array(self, new_array, weight_to_2, weight_to_1024):
        self._out_array = new_array
        self._weight0 = weight_to_2
        self._weight1 = weight_to_1024
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
        self.__input_neuro = 1024
        self.__output_neuro = 2

    def add_array(self, new_array, weight_to_1024, weiqht_to_2, iterations):
        self._out_array = new_array
        self._weight0 = weight_to_1024
        self._weight1 = weiqht_to_2
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
    def code(self):
        pass

    def decode(self):
        pass

    def learn(self):
        pass

    def set_ip(self):
        pass

# controller of neuronet
class Controller:
    def __init__(self):
        self.code = ModelCod()
        self.decode = ModelDecod()
        self.view = View()

    def specify_def(self):
        while 1:
            name_def = input()
            match name_def:
                case 'code':
                    full_path = input('provide full path')
                    array_byte = np.fromfile(full_path, dtype=np.byte)
                    self.code.add_array(array_byte, array_weight_2, array_weight_1024)
                case 'learn':
                    pass
                case 'set_ip':
                    pass
                case 'exit':
                    break
                case _:
                    print('there is no such def')