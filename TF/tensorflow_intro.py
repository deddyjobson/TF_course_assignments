import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

observations = 1000

xs = np.random.uniform(-10, 10, size=(observations, 1))
zs = np.random.uniform(-10, 10, size=(observations, 1))

generated_inputs = np.column_stack((xs, zs))
noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 2*xs - 3*zs + 5 + noise

# save data in a TF-friendly manner
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)


# load train data
training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1

# define the model
model = tf.keras.Sequential([
        tf.keras.layers.Dense(
                output_size,
                kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                bias_initializer=tf.random_uniform_initializer(-0.1, 0.1)
                )
        ])
model.compile(optimizer='sgd', loss='huber_loss')

# giving data to the model
model.fit(
            training_data['inputs'], training_data['targets'],
            epochs=100, verbose=0
            )

print('Time to get weights')
print(model.layers[0].get_weights())

if False:
    plt.plot(
            np.squeeze(model.predict_on_batch(training_data['inputs'])),
            np.squeeze(training_data['targets'])
            )
    plt.xlabel('outputs')
    plt.ylabel('targets')
    plt.show()
