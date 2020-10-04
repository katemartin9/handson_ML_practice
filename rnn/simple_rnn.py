import tensorflow as tf
from utils import *

# Create train and test sets
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]

# Baseline 1 naive forecasting, our prediction is the last value in the series
y_pred = X_valid[:, -1]
np.mean(tf.keras.losses.mean_squared_error(y_valid, y_pred))  # output 0.20...

fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0],
                y_label=("$x(t)$" if col==0 else None))
plt.savefig("time_series_plot.png")
plt.show()

# Baseline 2 - Simple Linear Model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=[50, 1]),
                                   tf.keras.layers.Dense(1)])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
model.evaluate(X_valid, y_valid)  # Output loss: 0.0046

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], n_steps, y_valid[0, 0], y_pred[0, 0])
plt.show()

# Implementing a simple RNN
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
optimizer = tf.keras.optimizers.Adam(lr=0.005)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
model.evaluate(X_valid, y_valid)  # Output 0.01

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], n_steps, y_valid[0, 0], y_pred[0, 0])
plt.show()

# Deep RNN
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20, return_sequences=True,
                              input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)  # allows to change output activation function
])
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
model.evaluate(X_valid, y_valid)  # Output 0.003
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], n_steps, y_valid[0, 0], y_pred[0, 0])
plt.show()

# Multiple steps ahead (one at a time)
steps_ahead = 10
series_new = generate_time_series(1, n_steps + steps_ahead)

X_new, Y_new = series_new[:, :n_steps], series_new[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
Y_pred = X[:, n_steps:]

plot_multiple_forecasts(X_new, Y_new, Y_pred)
plt.show()
