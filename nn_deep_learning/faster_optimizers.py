from tensorflow import keras
from functools import partial
import numpy as np


# exponential scheduling at each epoch
def exponential_decay(epoch, lr=0.01, s=20):
    return lr * 0.1**(epoch / s)


lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay)
K = keras.backend


# exponential decay at each iteration
class ExponentialDecay(keras.callbacks.Callback):
    def __init__(self, s=40000):
        super().__init__()
        self.s = s

    def on_batch_begin(self, batch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr*0.1**(1/self.s))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])
lr0 = 0.01
optimizer = keras.optimizers.Nadam(lr=lr0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
n_epochs = 25

# s = 20 * len(X_train) // 32 number of steps in 20 epochs (batch size = 32)
exp_decay = ExponentialDecay()

# Regularisation
RegularisedDense = partial(keras.layers.Dense,
                            activation='elu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularisedDense(300),
    RegularisedDense(100),
    RegularisedDense(10, activation='softmax',
                     kernel_initializer='glorot_uniform')
])


# Monte Carlo (MC) Dropout - provides better uncertainty estimates
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


# Max Norm regularisation
keras.layers.Dense(100, activation='elu',
                   kernel_initializer='he_normal',
                   kernel_constraint=keras.constraints.max_norm(1.))
