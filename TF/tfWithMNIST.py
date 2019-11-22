import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


mnist_dataset, mnist_info = tfds.load(name='mnist',
                                      with_info=True,
                                      as_supervised=True
                                      )

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = mnist_info.splits['train'].num_examples * 10 // 100
num_test_samples = mnist_info.splits['test'].num_examples


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label


scaled_train_and_validation_data = mnist_train.map(scale)
scaled_test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(
    BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(
    num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(
    num_validation_samples)

BATCH_SIZE = 64

# We batch the train data, but not the validation/test data.
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(BATCH_SIZE)
test_data = scaled_test_data.batch(BATCH_SIZE)

input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(
    # optimizer='adam',
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

NUM_EPOCHS = 13
model.fit(
    train_data,
    epochs = NUM_EPOCHS,
    validation_data=validation_data, 
    verbose = 1
)

model.evaluate(test_data)
