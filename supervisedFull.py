import keras.layers
from keras.models import Model


class SupervisedLearningModel(object):
    def __init__(self, input_shape):
        self.input_shape= input_shape
        self.model = self._build_model()

    def _build_model(self):
        x1 = keras.layers.Input(shape= self.input_shape)
        x2 = keras.layers.Dense(shape=64, activation="relu")(x1)
        x3 = keras.layers.BatchNormalization()(x2)
        x4 = keras.layers.Dense(shape=32, activation="relu")(x3)
        x5 = keras.layers.Dense(shape=1, activation="tanh")(x4)
        return Model(input=x1, output=x5)

    def __call__(self, *args, **kwargs):
        return self.model.predict_on_batch(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


