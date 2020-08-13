import tensorflow as tf
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib as mpl
from functools import partial

# load images
china = load_sample_image('china.jpg') / 255
flower = load_sample_image('flower.jpg') / 255
images = np.array([china, flower])
batch_size, height, weight, channels = images.shape  # (2, 427, 640, 3)

# create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2),
                   dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line in the middle
filters[3, :, :, 1] = 1  # horizontal line in the middle
outputs = tf.nn.conv2d(images, filters, strides=1,
                       padding='SAME')  # shape (2, 427, 640, 2)
plt.imshow(outputs[0, :, :, 1], cmap='gray')  # vertical temple
plt.axis('off')
plt.show()

plt.imshow(outputs[0, :, :, 0], cmap='gray')  # horizontal temple
plt.axis('off')
plt.show()

for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plt.subplot(2, 2,
                    image_index * 2 + feature_map_index + 1)
        plt.imshow(outputs[image_index, :, :, feature_map_index],
                   cmap='gray')
plt.axis('off')
plt.show()


def crop(images):
    return images[150:220, 130:250]


plt.imshow(crop(images[0, :, :, 0]), cmap='gray')

for feature_map_index, filename in enumerate(['china_vertical',
                                              "china_horizontal"]):
    plt.imshow(crop(outputs[0, :, :, feature_map_index]), cmap='gray')
    plt.show()

plt.imshow(filters[:, :, 0, 0])
plt.show()

plt.imshow(filters[:, :, 0, 1])
plt.show()

# Convolutional layer
conv = keras.layers.Conv2D(filters=32, kernel_size=3,
                           strides=1, padding='same',
                           activation='relu')  # 32 filters 3x3 with a step of 1

# Pooling layer (Max pooling)
max_pool = keras.layers.MaxPool2D(pool_size=2)
cropped_images = np.array([crop(image) for image in images],
                          dtype=np.float32)
output = max_pool(cropped_images)

fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2,
                           width_ratios=[2, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(cropped_images[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(output[0])  # plot the output for the 1st image
ax2.axis("off")
plt.show()


# Depth-wise pooling
class DepthMaxPool(keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding='VALID', **kwargs):
        super().__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        return tf.nn.max_pool(inputs,
                              ksize=(1, 1, 1,self.pool_size),
                              strides=(1, 1, 1, self.pool_size),
                              padding=self.padding)


depth_pool = DepthMaxPool(3)
depth_output = depth_pool(cropped_images)
depth_output.shape

# with Lambda
depth_pool2 = keras.layers.Lambda(lambda x: tf.nn.max_pool(x,
                                                           ksize=(1, 1, 1, 3),
                                                           strides=(1, 1, 1, 3),
                                                           padding='VALID'))
depth_output2 = depth_pool2(cropped_images)
depth_output2.shape

# ResNet-34
DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, strides=1,
                        padding='SAME', use_bias=False)


class ResidualUnit(keras.layers.Layer):
    """Residual unit layer"""
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        # creating all layers needed
        self.main_layer = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()
        ]
        # creating skip layers
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layer:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        # combining outputs from the main layers and skip layers (if any)
        return self.activation(Z + skip_Z)


model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                        input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2,
                                 padding='SAME'))
prev_filters = 64
# adding residual layers to the model
# the first 3 - 64 filters
# next 4 - 128 filters
# next 6 - 256 filters
# last 3 - 512 filters
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    print(filters, prev_filters, strides)
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

# load pre-trained ResNet, expects 224 x 224
model = keras.applications.resnet50.ResNet50(weights='imagenet')
images_resized = tf.image.resize(images, [224, 224])
china_box = [0, 0.03, 1, 0.68]
flower_box = [0.19, 0.26, 0.86, 0.7]
images_resized2 = tf.image.crop_and_resize(images, [china_box, flower_box], [0, 1], [224, 224])
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
y_proba = model.predict(inputs)

top_k = keras.applications.resnet50.decode_predictions(y_proba, top=3)
for image_idx in range(len(images)):
    print(f'Image {image_idx}')
    for class_id, name, y_proba in top_k[image_idx]:
        print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
        print()