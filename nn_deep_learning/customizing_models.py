import tensorflow as tf
from tensorflow import keras


# LOSS FNs
# iplementing huber loss fn
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)


# model.compile(loss=huber_fn, optimizer='nadam)

# Saving and loading model with custom components
model = keras.models.load_model('my_mnist_model.h5',
                                custom_objects={'huber_fn': huber_fn})


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


model.compile(loss=create_huber(2.0), optimizer='nadam')
model = keras.models.load_model('my_mnist_model.h5',
                                custom_objects={'huber_fn': create_huber(2.0)})


# don't need to specify threshold when loading
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# model.compile(loss=HuberLoss(2.), optimizer='nadam)
# this will load a json config file
model2 = keras.models.load_model('my_mnist_model.h5', custom_objects={'HuberLoss': HuberLoss})


# ACTIVATION, REG FNS
def my_softplus(z):
    """Same as
    keras.activations.softplus()
    tf.nn.softplus()
    """
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    """keras.initializers.glorot_normal()
    Draws samples from a truncated normal distribution centered on 0
    """
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, sttdev=stddev, dtype=dtype)


def my_l1_reg(weights):
    """keras.regularizes.l1
    absolute sum of weights * reg factor
    """
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):
    """Custom constraint that ensure all weights are positive
    keras.constraints.noneg() or tf.nn.relu()
    """
    return tf.where(weights < 0, tf.zeros_like(weights), weights)


layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_reg,
                           kernel_constraint=my_positive_weights)


# METRICS
class HuberLossMetric(keras.metrics.Metric):

    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weights=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# LAYERS
# custom layer with no weights e.g Flatten() or ReLU()
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))


# custom layer with weights
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=[batch_input_shape[-1], self.units],
            initializer='glorot_normal'
        )
        self.bias = self.add_weight(
            name='bias',
            shape=[self.units],
            initializer='zeros'
        )
        super().build(batch_input_shape)

    def call(self, X):
        """multiplying weights with input and adding bias"""
        return self.activation(X @ self.kernel) + self.bias

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}


# MODEL
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons,
                                          activation='elu',
                                          kernel_initializer='he_normal')
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z


class ResidualRegressor(keras.models.Model):

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation='elu',
                                          kernel_initializer='he_normal')
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)


class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30,
                                         activation='selu',
                                         kernel_initializer='lecun_normal')
                      for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(
            reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)


# GRADIENTS AUTODIFF
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2


# derivative wrt w1 = 6*w1 + 2*w2
# derivative wrt w2 = 2*w1
# (w1, w2) = (5, 3)
w1, w2 = 5, 3
eps = 1e-6
(f(w1 + eps, w2) - f(w1, w2)) / eps
(f(w1, w2 + eps) - f(w1, w2)) / eps

# using autodiff
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)
gradients = tape.gradient(z, [w1, w2])
# calling grad more than once
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)
del tape


# TF FUNCTION AND GRAPHS
def cube(x):
    return x ** 3


tf_cube = tf.function(cube)
tf_cube(2)  # returns tensors


# OR use a decorator
@tf.function
def tf_cube(x):
    return x ** 3


@tf.function
def add_10(x):
    condition = lambda i, x: tf.less(i, 10)
    body = lambda i, x: (tf.add(i, 1), tf.add(x, 1))
    final_i, final_x = tf.while_loop(condition, body,
                                     [tf.constant(0), x])
    return final_x


add_10(tf.constant(5))
