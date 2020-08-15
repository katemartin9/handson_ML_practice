import tensorflow as tf
from tensorflow import keras
import numpy as np

np.random.seed(42)

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)

for item in dataset:
    print(item)

# Data Transformations
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

dataset.map(lambda x: x * 2)
dataset.apply(tf.data.experimental.unbatch())
dataset.filter(lambda x: x < 10)
for item in dataset.take(3):
    print(item)

# shuffling
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
for item in dataset:
    print(item)


