from tensorflow import keras
import numpy as np
import tensorflow as tf

shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)  # char level encoding rather than word

# mapping every token to a character id (1 to n of distinct characters)
tokenizer.fit_on_texts(shakespeare_text)
tokenizer.texts_to_sequences(['First'])
tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])  # Output ['f i r s t']
max_id = len(tokenizer.word_index)  # distinct n of chars
dataset_size = tokenizer.document_count  # total n of chars

# encoding full text and subtracting one to start from 0
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
# splitting dataset
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps = 100  # sequence length
window_length = n_steps + 1

# window() creates non-overlapping windows each containing 101 chars
dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

np.random.seed(42)
tf.random.set_seed(42)

batch_size = 32  # number of windows in a batch
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)

for X_batch, Y_batch in dataset.take(1):
    print(X_batch.shape, Y_batch.shape)

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True,
                     input_shape=[None, max_id],
                     dropout=0.2,
                     recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     dropout=0.2,
                     recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                    activation='softmax'))
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam')
history = model.fit(dataset, steps_per_epoch=train_size // batch_size,
                    epochs=10)


def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)