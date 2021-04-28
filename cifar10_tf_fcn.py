import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.vis_utils import plot_model

(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

# Data set labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot 9 random images with labels
idxs = random.sample(range(x_train_original.shape[0]), 9)
fig, ax = plt.subplots(3, 3, figsize=(10, 10), dpi=100)
for i in range(3):
    for j in range(3):
        index = idxs[j + i * 3]
        ax[i, j].imshow(x_train_original[index])
        y_true_label = y_train_original[index][0]
        ax[i, j].set_title(f'{labels[y_true_label]}')

plt.tight_layout()
plt.show()

# Check if dataset is balanced
unique, counts = np.unique(y_train_original, return_counts=True)
print(unique, counts)

# labels to categorical (one-hot encoding)
y_train = np_utils.to_categorical(y_train_original, 10)
y_test = np_utils.to_categorical(y_test_original, 10)

# values in the interval 0 - 1
x_train = x_train_original / 255
x_test = x_test_original / 255

# Hyperparameters
learning_rate = 0.001
n_epochs = 50
batch_size = 32
validation_split = 0.2
activation = 'relu'

# Model definition
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3), name="input_layer"))
model.add(Dense(units=1024, activation=activation, name="hidden_layer_1"))
model.add(Dense(units=512, activation=activation, name="hidden_layer_2"))
model.add(Dense(units=256, activation=activation, name="hidden_layer_3"))
model.add(Dense(units=10, activation='softmax', name="output_layer"))

# Model summary and graph
model.summary()
plot_model(model, to_file=os.path.join('images', 'cifar10_tf_fcn.png'), show_shapes=True)

# Optimizer (https://keras.io/api/optimizers/)
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

# Loss (https://keras.io/api/losses/)
loss = keras.losses.CategoricalCrossentropy()

# Compile the model before training
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

# uncomment to use Tensorboard
# tensorboard_callback = TensorBoard(log_dir='.', histogram_freq=1)

history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs, verbose=1,
                    validation_split=validation_split,
                    shuffle=True,
                    # uncomment to use Tensorboard  (tensorboard --logdir .)
                    # callbacks=[tensorboard_callback]
                    )

# Plot training
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
ax[0].plot(history.history['acc'], 'r')
ax[0].plot(history.history['val_acc'], 'g')
ax[0].set_xlabel("Num of Epochs")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Training Accuracy vs Validation Accuracy")
ax[0].legend(['train', 'validation'])

ax[1].plot(history.history['loss'], 'r')
ax[1].plot(history.history['val_loss'], 'g')
ax[1].set_xlabel("Num of Epochs")
ax[1].set_ylabel("Loss")
ax[1].set_title("Training Loss vs Validation Loss")
ax[1].legend(['train', 'validation'])

plt.tight_layout()
plt.show()

# Final test accuracy
evaluation = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
print(f'test accuracy: {evaluation[1]:.4f}')

# Plot 9 random predictions
idxs = random.sample(range(x_test.shape[0]), 9)
y_hat = model.predict(x_test[idxs])
fig, ax = plt.subplots(3, 3, figsize=(10, 10), dpi=100)
for i in range(3):
    for j in range(3):
        index = idxs[j + i * 3]
        ax[i, j].imshow(x_test[index])
        y_pred_label = np.argmax(y_hat[j + i * 3])
        y_true_label = np.argmax(y_test[index])

        ax[i, j].set_title(f'True: {labels[y_true_label]}, predicted: {labels[y_pred_label]}')

plt.tight_layout()
plt.show()



