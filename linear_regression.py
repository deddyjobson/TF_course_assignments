import numpy as np
import matplotlib.pyplot as plt


observations = 1000
xs = np.random.uniform(-10, 10, (observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

inputs = np.column_stack((xs, zs))

print(inputs.shape)

noise = np.random.uniform(-1, 1, (observations, 1))

targets = 2*xs - 3*zs + 5 + noise

plt.scatter(xs, targets)
plt.show()

# initialize parameters of the model
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

eta = 0.01

for x, z, y in np.column_stack((inputs, targets)):
    # make a prediction
    pred = w1 * x + w2 * z + b

    # update the values
    w1 -= eta*(pred - y)*x
    w2 -= eta*(pred - y)*z
    b -= eta*(pred - y)

print(w1, w2, b)
