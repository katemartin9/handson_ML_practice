import numpy as np
from tensorflow import keras
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_valid, X_train = X_train[:5000] / 255.0, X_train[5000:] / 255.0
y_valid, y_train = y_train[:5000], y_train[5000:]


def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

tf.random.set_seed(42)
np.random.seed(42)
# MODEL A
model_A = keras.models.Sequential()
model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_A.add(keras.layers.Dense(n_hidden, activation='selu'))
model_A.add(keras.layers.Dense(8, activation='softmax'))
model_A.compile(loss='sparse_categorical_crossentropy',
                optimizer=keras.optimizers.SGD(lr=0.001),
                metrics=['accuracy'])
history_a = model_A.fit(X_train_A, y_train_A,
                      epochs=20,
                      validation_data=(X_valid_A, y_valid_A))
model_A.save('my_model_A.h5')
# MODEL B
model_B = keras.models.Sequential()
model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_B.add(keras.layers.Dense(n_hidden, activation='selu'))
model_B.add(keras.layers.Dense(1, activation='sigmoid'))
model_B.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.SGD(lr=0.001),
                metrics=['accuracy'])
history_b = model_B.fit(X_train_B, y_train_B, epochs=20,
                      validation_data=(X_valid_B, y_valid_B))

# TRANSFER LEARNING
model_A = keras.models.load_model('my_model_A.h5')
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation='sigmoid'))

model_A_clone = keras.models.clone_model(model_A)  # so model A is detached from B
model_A_clone.set_weights(model_A.get_weights())
# freeze the reused layers
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss='binary_crossentropy', optimizer='sgd',
                     metrics=['accuracy'])
# train the model for a few epochs and then unfreeze the reused layers
history = model_B_on_A.fit(X_train_B, y_train_B,
                           epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
history = model_B_on_A.fit(X_train_B, y_train_B,
                           epochs=16,
                           validation_data=(X_valid_B, y_valid_B))
