import keras.layers
from keras.models import Model


class SupervisedLearningModel(object):
    def __init__(self, input_shape):
        self.input_shape= input_shape
        self.model = self._build_model()
        self.model.summary()
        self.model.compile(optimizer="Adam", loss="mse")

    def _build_model(self):
        x1 = keras.layers.Input(shape= (self.input_shape, ))
        x4 = keras.layers.Dense(64, activation="relu", kernel_initializer="glorot_normal")(x1)
        x5 = keras.layers.Dense(12, activation="relu", kernel_initializer="glorot_normal")(x4)
        x7 = keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_normal")(x5)
        return Model(input=x1, output=x7)

    def __call__(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


import numpy as np
if __name__ =="__main__":
    ydata = np.load("appliances.npy")
    xdata2 = np.load("lights.npy")
    xdata = np.load("X_data.npy")

    xdata = np.hstack((xdata, xdata2))

    test_y = ydata[-100: ]
    test_x = xdata[-100: ]

    train_x = xdata[: -100]
    train_y = ydata[: -100]

    model = SupervisedLearningModel(xdata.shape[1])

    model.model.fit(train_x, train_y, epochs=5, batch_size=16)
    for i in range(100):
        print("original", test_y[i])
        print("infer", model(np.expand_dims(test_x[i], 1).T))

    print(model.model.evaluate(test_x, test_y))