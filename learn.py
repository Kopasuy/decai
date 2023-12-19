import numpy as np

def create_array(size):
    array = np.array([])
    for i in range(size):
        array = np.append(array, 0)
    return array

def create_array2D(size1, size2):
    array = np.array([])
    for i in range(size1):
        array_dop = create_array(size2)
        array = np.append(array, array_dop)
    return array

def learn(min_lim, max_lim, steps, dropout_inp, array_inp, array_weight, array_out, array_true_pred):
    # determine the required values
    size_exp = len(array_true_pred)
    step = (abs(min_lim) + abs(max_lim)) / steps
    pred = array_inp @ array_weight

    w = create_array(steps)
    for i in range(steps):
        w[i] = min_lim + step * i

    error = create_array2D(2, size_exp)
    error_usr = create_array2D(2, steps)

    array_inp_dropout = array_inp[dropout_inp]
    array_inp[dropout_inp] = create_array(size_exp)

    n_weight = 0
    for weight in w:
        for n_exper in range(size_exp):
            array_out_sum = np.sum(array_out[n_exper]) / 2
            error[0][n_exper] = (np.exp(array_inp_dropout[n_exper] * weight + pred[0][n_exper]) - array_out_sum +
                                 array_true_pred[0][n_exper]) ** 2
            error[1][n_exper] = (np.exp(array_inp_dropout[n_exper] * weight + pred[1][n_exper]) - array_out_sum +
                                 array_true_pred[1][n_exper]) ** 2
        error_usr[0][n_weight] = np.sum(error[0]) / len(error[0])
        error_usr[1][n_weight] = np.sum(error[1]) / len(error[1])
        n_weight += 1