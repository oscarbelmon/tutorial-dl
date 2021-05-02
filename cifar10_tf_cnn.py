import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.vis_utils import plot_model


save_plots = True

(x_tr_orig, y_tr_orig), (x_ts_orig, y_ts_orig) = cifar10.load_data()

# Data set labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot 9 random images with labels
idxs = random.sample(range(x_tr_orig.shape[0]), 9)
fig, ax = plt.subplots(3, 3, figsize=(10, 10), dpi=100)
for i in range(3):
    for j in range(3):
        index = idxs[j + i * 3]
        ax[i, j].imshow(x_tr_orig[index])
        y_true_label = y_tr_orig[index][0]
        ax[i, j].set_title(f'{labels[y_true_label]}')

plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('images', 'cifar10_tf_cnn_train.png'))
else:
    plt.show()

# Check if dataset is balanced
unique, counts = np.unique(y_tr_orig, return_counts=True)
print(unique, counts)

# Hyperparameters
learning_rate = 0.001
n_epochs = 1000
batch_size = 64
validation_split = 0.1
activation = 'relu'
dropout = 0.5

# use data augmentation?
data_augmentation = True

# split training into training and validation
x_tr_orig, x_vl_orig, y_tr_orig, y_vl_orig = train_test_split(x_tr_orig, y_tr_orig, test_size=validation_split)

# labels to categorical (one-hot encoding)
y_train = np_utils.to_categorical(y_tr_orig, 10)
y_val = np_utils.to_categorical(y_vl_orig, 10)
y_test = np_utils.to_categorical(y_ts_orig, 10)

# normalization to 0 mean and 1 std
mean = np.mean(x_tr_orig)
std = np.std(x_tr_orig)
x_train = (x_tr_orig - mean) / std
x_val = (x_vl_orig - mean) / std
x_test = (x_ts_orig - mean) / std

# Model definition
model = Sequential()
model.add(Conv2D(input_shape=(32, 32, 3), kernel_size=(2, 2), padding='same', strides=(2, 2), filters=32))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
model.add(Conv2D(kernel_size=(2, 2), padding='same', strides=(2, 2), filters=64))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
model.add(Flatten())
model.add(Dense(256, activation=activation))
model.add(Dropout(dropout))
model.add(Dense(128, activation=activation))
model.add(Dropout(dropout))
model.add(Dense(10, activation='softmax'))

# Model summary and graph
model.summary()
plot_model(model, to_file=os.path.join('images', 'cifar10_tf_cnn.png'), show_shapes=True)

# Optimizer (https://keras.io/api/optimizers/)
optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)

# Loss (https://keras.io/api/losses/)
loss = keras.losses.CategoricalCrossentropy()

# Compile the model before training
model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

# uncomment to use Tensorboard
# tensorboard_callback = TensorBoard(log_dir='.', histogram_freq=1)

if data_augmentation:
    # set up image augmentation
    datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(x_train)
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=n_epochs, verbose=1,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        # uncomment to use Tensorboard  (tensorboard --logdir .)
                        # callbacks=[tensorboard_callback]
                        )
else:
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=n_epochs, verbose=1,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        # uncomment to use Tensorboard  (tensorboard --logdir .)
                        # callbacks=[tensorboard_callback]
                        )

# Plot training history
_, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
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
if save_plots:
    plt.savefig(os.path.join('images', f'cifar10_tf_cnn_history_{"dat" if data_augmentation else "daf"}.png'))
else:
    plt.show()

# Final test accuracy
evaluation = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
print(f'test accuracy: {evaluation[1]:.4f}')

# Plot 9 random predictions
idxs = random.sample(range(x_test.shape[0]), 9)
y_hat = model.predict(x_test[idxs])
_, ax = plt.subplots(3, 3, figsize=(10, 10), dpi=100)
for i in range(3):
    for j in range(3):
        index = idxs[j + i * 3]
        ax[i, j].imshow(x_ts_orig[index])
        y_pred_label = np.argmax(y_hat[j + i * 3])
        y_true_label = np.argmax(y_test[index])
        ax[i, j].set_title(f'True: {labels[y_true_label]}, predicted: {labels[y_pred_label]}')

plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('images', 'cifar10_tf_cnn_test.png'))
else:
    plt.show()
