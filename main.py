import numpy as np

# layers
size_1B = 8  # bit
size_100B = 800  # bit
size_100KB = 819_200  # bit
size_100MB = 838_860_800  # bit
size_1GB = 8_589_934_592  # bit

# 100B 2 1B and vice versa
weight0_code = np.random.randn(size_100B, size_1B)
weight0_encode = np.random.randn(size_1B, size_100B)

# 100MB 2 100B and vice versa
weight1_code = np.random.randn(size_100KB, size_100B)
weight1_encode = np.random.randn(size_100B, size_100KB)

# 100MB 2 100KB and vice versa
weight2_code = np.random.randn(size_100MB, size_100KB)
weight2_encode = np.random.randn(size_100KB, size_100MB)

# 1GB 2 100MB and vice versa
weight3_code = np.random.randn(size_1GB, size_100MB)
weight3_encode = np.random.randn(size_100MB, size_1GB)

def create_array(size):
    array = []
    for i in range(size):
        array.append(0)
    return array

def byte_2_int(array, size):
    value2int = int(size/size_1GB)
    array_int = create_array(size_1GB)
    array_int_dop = create_array(value2int)
    j_stat = 0
    for i in range(size_1GB):
        for j in range(value2int):
            array_int_dop[j] = array[j_stat + j]
            j_stat += value2int
        array_int[i] = int(''.join(map(str,array_int_dop)), value2int)
    return array_int

def identify_layers(size):
    if size <= size_100B:
        return 0
    elif size <= size_100KB:
        return 1
    elif size <= size_100MB:
        return 2
    elif size <= size_1GB:
        return 3

def binary_step(array, step):  # binary step function
    for i in range(len(array)):
        if array[i] >= step:
            array[i] = 1
        else:
            array[i] = 0
    return array

def predict(byte_array, compression):
    size = len(byte_array)
    lvl = 0
    if size > size_1GB:
        byte_array = byte_2_int(byte_array, size)
        lvl = 3
    else:
        lvl = identify_layers(size)
    differ = lvl - compression
    if differ < 0:
        print('Compression should be less')
    else:
        w = 0
        while differ >= 0:
            match lvl:
                case 3:
                    w = weight3_code
                case 2:
                    w = weight2_code
                case 1:
                    w = weight1_code
                case 0:
                    w = weight0_code
            byte_array = np.dot(byte_array, w)
            byte_array = binary_step(byte_array,0)
            lvl -= 1
            differ -= 1
    return byte_array