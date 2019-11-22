import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# reading data
data = np.loadtxt(open("data/Audiobooks-data.csv", "rb"),
                  delimiter=",", skiprows=0)
np.random.shuffle(data)

# obtaining validation and test data
X_train, X_test, y_train, y_test = train_test_split(
    data[:, 1:-1], data[:, -1], test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

# standardizing data
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# data analysis (on train data)
num_pos = np.sum(y_train)  # number of people who convert
print("Class distribution:")
print("Don't convert (0):{0}\tDo convert (1):{1}".format(
    1-num_pos, num_pos))

BATCH_SIZE = 64

input_size = X_train.shape[1]
output_size = 1
hidden_layer_size = 80

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

NUM_EPOCHS = 100
model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(X_val, y_val),
    verbose=2
)

# now we evaluate model on untouched test data
y_pred = model.predict(X_test)
print(tf.math.confusion_matrix(y_test, y_pred))

num_pos = np.sum(y_test > 0)  # number of people who convert
print("Test class distribution:")
print("Don't convert (0):{0}\tDo convert (1):{1}".format(
    y_test.shape[0]-num_pos, num_pos))

print("\nEvaluation over unbalanced data...")
# generate roc curve
fpr, tpr, _ = roc_curve(y_test.reshape(-1), y_pred.reshape(-1))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating Characteristic - AudioBook')
plt.legend(loc="lower right")
plt.savefig("audioBookROC.png")
plt.show()

# model.evaluate(X_test,y_test)
