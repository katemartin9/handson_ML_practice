import os
from pathlib import Path
from utils import load_files
from synthesizer import play_chords
import tensorflow as tf
from tensorflow import keras
from IPython.display import Audio

path = os.getcwd()
jsb_chorales_dir = Path(os.path.join(os.getcwd(), 'jsb_chorales'))

train_files = sorted(jsb_chorales_dir.glob("train/chorale_*.csv"))
valid_files = sorted(jsb_chorales_dir.glob("valid/chorale_*.csv"))
test_files = sorted(jsb_chorales_dir.glob("test/chorale_*.csv"))

train_chorales = load_files(train_files)
valid_chorales = load_files(valid_files)
test_chorales = load_files(test_files)


def extract_unique_chorcales(chorcales):
    notes = set()
    for chorcales in chorcales:
        for chorcale in chorcales:
            for chord in chorcale:
                notes |= set(chord)
    return notes


notes = extract_unique_chorcales((train_chorales, valid_chorales, test_chorales))
n_notes = len(notes)
min_note = min(notes - {0})
max_note = max(notes)

assert min_note == 36
assert max_note == 81

# for index in range(3):
# play_chords(train_chorales[index])


# Model that predicts the next cord given all the previous chords
def create_target(batch):
    X = batch[:, :-1]
    Y = batch[:, 1:]  # predict next note in each arpegio, at each step
    return X, Y


def preprocess(window):
    window = tf.where(window == 0, window, window - min_note + 1) # shift values
    return tf.reshape(window, [-1])  # convert to arpegio


def bach_dataset(chorales, batch_size=32, shuffle_buffer_size=None,
                  window_size=32, window_shift=16, cache=True):

    def batch_window(window):
        return window.batch(window_size + 1)

    def to_window(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(window_size + 1, window_shift,
                                 drop_remainder=True)
        return dataset.flat_map(batch_window)

    chorales = tf.ragged.constant(chorales, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(chorales)
    dataset = dataset.flat_map(to_window).map(preprocess)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(create_target)
    return dataset.prefetch(1)


train_set = bach_dataset(train_chorales, shuffle_buffer_size=1000)
valid_set = bach_dataset(valid_chorales)
test_set = bach_dataset(test_chorales)

n_embedding_dims = 5
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=n_notes,
                           output_dim=n_embedding_dims,
                           input_shape=[None]),
    keras.layers.Conv1D(32, kernel_size=2,
                        padding="causal",
                        activation="relu"),
    keras.layers.BatchNormalization(),  # for faster better convergence
    keras.layers.Conv1D(48, kernel_size=2,
                        padding="causal",
                        activation='relu',
                        dilation_rate=2),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(64, kernel_size=2,
                        padding='causal',
                        activation='relu',
                        dilation_rate=4),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(96, kernel_size=2,
                        padding="causal",
                        activation="relu",
                        dilation_rate=8),
    keras.layers.BatchNormalization(),
    keras.layers.LSTM(256, return_sequences=True),  # catching long-term patterns
    keras.layers.Dense(n_notes, activation="softmax")
])

model.summary()

optimizer = keras.optimizers.Nadam(lr=1e-3)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(train_set, epochs=20, validation_data=valid_set)

model.save('my_back_model.h5')
model.evaluate(test_set)