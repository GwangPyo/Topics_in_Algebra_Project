import numpy as np


def to_appliance(x):
    appliacne_data = x
    appliacne_data *= 0.38950541957475976
    appliacne_data += -0.016425521817009823
    appliacne_data = np.arctanh(appliacne_data)
    appliacne_data *= 100
    appliacne_data += 81.24246262984545
    return appliacne_data


def sampling(size, validation_size=100):
    """
    :param size: the size of data
    :return: tuple of indices one for validation and training
    """
    indexset = list(range(size))
    np.random.shuffle(indexset)
    validation = indexset[:validation_size]
    training = indexset[validation_size:]
    return np.array(validation), np.array(training)


def read_sample_number():
    with open("samplenumber.txt", "r") as f:
        samplenumber = int(f.readline())
    return samplenumber


def commit(samplenumber):
    with open("samplenumber.txt", "w") as f:
        f.write(str(samplenumber + 1))

