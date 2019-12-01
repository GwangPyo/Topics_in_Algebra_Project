import keras.layers
import keras.optimizers
from keras.models import Model
import numpy as np
import pandas as pd
from utils import to_appliance, sampling, read_sample_number, commit


class SupervisedLearningModel(object):
    def __init__(self, input_shape):
        self.input_shape= input_shape
        self.model = self._build_model()
        self.model.summary()
        optimizer = keras.optimizers.Adam(lr=1.5e-3)
        self.model.compile(optimizer=optimizer, loss="mse")

    def _build_model(self):

        x1 = keras.layers.Input(shape= (self.input_shape, ))
        x2 = keras.layers.BatchNormalization(momentum=0.8)(x1)
        x4 = keras.layers.Dense(128, activation="relu", kernel_initializer="glorot_normal")(x2)
        x3 = keras.layers.BatchNormalization(momentum=0.8)(x4)
        x5 = keras.layers.Dense(64, activation="relu", kernel_initializer="glorot_normal")(x3)
        x6 = keras.layers.BatchNormalization(momentum=0.8)(x5)
        x7 = keras.layers.Dense(32, activation="relu", kernel_initializer="glorot_normal")(x6)
        x8 = keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_normal")(x7)
        return Model(input=x1, output=x8)

    def __call__(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


if __name__ =="__main__":
    ydata = np.load("appliances.npy")
    print(np.max(ydata))
    print(np.min(ydata))
    xdata2 = np.load("lights.npy")
    xdata = np.load("X_data.npy")

    xdata = np.hstack((xdata, xdata2))

    for _ in range(1):
        validation, training = sampling(size=len(xdata), validation_size=1000)
        test_y = ydata[validation, :]
        test_x = xdata[validation, : ]

        train_x = xdata[training, :]
        train_y = ydata[training, :]

        model = SupervisedLearningModel(xdata.shape[1])

        diff = []
        model.model.fit(train_x, train_y, epochs=50, batch_size=16)
        record =np.zeros((1000, 3))
        for i in range(1000):
            original_one = test_y[i][0]
            inferred = model(np.expand_dims(test_x[i], 1).T).flatten()[0]
            original_one= to_appliance(original_one)
            inferred = np.round(to_appliance(inferred), -1)
            if inferred >= 100:
                inferred = 100

            if original_one >= 100:
                original_one = 100

            diff.append(abs(original_one- inferred))
            print("original", original_one)
            print("infer", inferred)

            record[i, 0] = inferred
            record[i, 1] = original_one
            record[i, 2] = abs(original_one - inferred)


        print(model.model.evaluate(test_x, test_y))

        samplenumber = read_sample_number()
        df = pd.DataFrame(record, columns=["inferrence", "original", "diff"])
        df.to_csv("records/accuracy{}.csv".format(samplenumber).format(samplenumber), header=True, index=False )
        commit(samplenumber)
        model.model.save_weights("model.h5")