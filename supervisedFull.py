import keras.layers
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SupervisedLearningModel(object):
    def __init__(self, input_shape):
        self.input_shape= input_shape
        self.model = self._build_model()
        self.model.summary()
        self.model.compile(optimizer="Adam", loss="mse")

    def _build_model(self):

        x1 = keras.layers.Input(shape= (self.input_shape, ))
        x4 = keras.layers.Dense(64, activation="relu", kernel_initializer="glorot_normal")(x1)
        x3 = keras.layers.BatchNormalization()(x4)
        x5 = keras.layers.Dense(12, activation="relu", kernel_initializer="glorot_normal")(x3)
        x7 = keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_normal")(x5)
        return Model(input=x1, output=x7)

    def __call__(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


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


def to_appliance(x):
    x = np.arctanh(x)
    MEAN = 9.76949581960983
    DEV = 10.252229296483618
    x *= DEV
    x += MEAN
    return x * 10


def read_sample_number():
    with open("samplenumber.txt", "r") as f:
        samplenumber = int(f.readline())
    return samplenumber


def commit(samplenumber):
    with open("samplenumber.txt", "w") as f:
        f.write(str(samplenumber))



if __name__ =="__main__":
    ydata = np.load("appliances.npy")
    xdata2 = np.load("lights.npy")
    xdata = np.load("X_data.npy")

    print(xdata.shape)
    print(xdata2.shape)
    xdata = np.hstack((xdata, xdata2))

    validation, training = sampling(size=len(xdata))
    test_y = ydata[validation, :]
    test_x = xdata[validation, : ]

    train_x = xdata[training, :]
    train_y = ydata[training, :]

    model = SupervisedLearningModel(xdata.shape[1])

    diff = []
    model.model.fit(train_x, train_y, epochs=5, batch_size=32)
    record =np.zeros((100, 3))
    for i in range(100):
        original_one = test_y[i][0]
        inferred = model(np.expand_dims(test_x[i], 1).T).flatten()[0]
        original_one= to_appliance(original_one)
        inferred = to_appliance(inferred)

        diff.append(abs(original_one- inferred))
        print("original", original_one)
        print("infer", inferred)

        record[i, 0] = inferred
        record[i, 1] = original_one
        record[i, 2] = abs(original_one - inferred)

    plt.hist(diff, bins=100)
    plt.show()
    print(model.model.evaluate(test_x, test_y))