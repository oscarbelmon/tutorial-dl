import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.utils.vis_utils import plot_model

save_plots = False
train_model = True

# download dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud/data
df = pd.read_csv(os.path.join('data', 'fraud', 'creditcard.csv'))

# normalize columns Time & Amount
df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

# check counts for each class
print('Dataset class distribution')
print(pd.value_counts(df['Class'], sort=True))

# training set will be composed only of normal transactions
df_normal = df[df.Class == 0]
df_fraud = df[df.Class == 1]
df_train, df_test = train_test_split(df_normal, test_size=0.2, random_state=42)
df_test = df_test.append(df_fraud).sample(frac=1).reset_index(drop=True)
df_val, df_test = train_test_split(df_test, test_size=0.5, stratify=df_test['Class'], random_state=42)

print('Training set class distribution')
print(pd.value_counts(df_train['Class'], sort=True))
print('Validation set class distribution')
print(pd.value_counts(df_val['Class'], sort=True))
print('Test set class distribution')
print(pd.value_counts(df_test['Class'], sort=True))

x_train = df_train.drop(['Class'], axis=1).values
y_val = df_val['Class'].values
x_val = df_val.drop(['Class'], axis=1).values
y_test = df_test['Class'].values
x_test = df_test.drop(['Class'], axis=1).values

if train_model:
    n_epochs = 200
    batch_size = 128
    input_dim = x_train.shape[1]
    activation = 'relu'
    learning_rate = 0.01

    autoencoder = Sequential()

    # https://keras.io/api/layers/regularizers/
    autoencoder.add(Dense(input_shape=(input_dim, ), units=24, activation=activation,
                    activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dense(20, activation=activation))
    autoencoder.add(Dense(16, activation=activation))
    autoencoder.add(Dense(12, activation=activation))
    autoencoder.add(Dense(8, activation=activation))
    autoencoder.add(Dense(12, activation=activation))
    autoencoder.add(Dense(16, activation=activation))
    autoencoder.add(Dense(20, activation=activation))
    autoencoder.add(Dense(24, activation=activation))
    autoencoder.add(Dense(input_dim, activation=activation))

    # Optimizer (https://keras.io/api/optimizers/)
    optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)

    # Loss (https://keras.io/api/losses/)
    loss = keras.losses.mean_squared_error

    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()
    plot_model(autoencoder, to_file=os.path.join('images', 'fraud_detection_tf_ae.png'), show_shapes=True)

    # saves the best model so far
    check_point = ModelCheckpoint(filepath='fraud_detection.h5', save_best_only=True, verbose=0)

    # stops training if validation loss is not improving
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')

    history = autoencoder.fit(x_train, x_train, epochs=n_epochs, batch_size=batch_size, shuffle=True,
                              validation_data=(x_val[y_val == 0], x_val[y_val == 0]), verbose=1,
                              callbacks=[check_point, early_stop])

    # Plot training history
    _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    ax.plot(history.history['loss'], 'r')
    ax.plot(history.history['val_loss'], 'g')
    ax.set_xlabel('Num of Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss vs Validation Loss')
    ax.legend(['train', 'validation'])
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join('images', 'fraud_detection_tf_ae_history.png'))
    else:
        plt.show()

autoencoder = load_model('fraud_detection.h5')
x_val_hat = autoencoder.predict(x_val)
mse = np.mean(np.power(x_val - x_val_hat, 2), axis=1)
error_df = pd.DataFrame({'rec_error': mse, 'true_class': y_val})
error_fraud = error_df.loc[(error_df['true_class'] == 1)]['rec_error'].values
error_no_fraud = error_df.loc[(error_df['true_class'] == 0)]['rec_error'].values

_, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
ax.hist(error_no_fraud, bins=2000, alpha=0.75, density=True, label='no fraud')
ax.hist(error_fraud, bins=2000, alpha=0.75, density=True, label='fraud')
plt.tight_layout()
plt.legend()
if save_plots:
    plt.savefig(os.path.join('images', 'fraud_detection_tf_ae_histogram.png'))
else:
    plt.show()

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.true_class, error_df.rec_error)
f1 = 2 * precision_rt[1:] * recall_rt[1:] / (precision_rt[1:] + recall_rt[1:])

_, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
ax.plot(threshold_rt, precision_rt[1:], label='Precision', linewidth=2)
ax.plot(threshold_rt, recall_rt[1:], label='Recall', linewidth=2)
ax.plot(threshold_rt, f1, label="f-1", linewidth=2, linestyle='--', alpha=0.5)
ax.set_title('Precision, recall anf f-1 for different threshold values')
ax.set_xlabel('Threshold')
ax.set_ylabel('Precision/Recall/F-1')
plt.legend()
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('images', 'fraud_detection_prec_rec.png'))
else:
    plt.show()


threshold_fixed = 5
groups = error_df.groupby('true_class')
_, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
for name, group in groups:
    ax.plot(group.index, group.rec_error, marker='o', ms=2.5, linestyle='', alpha=0.75,
            label='Fraud' if name == 1 else 'Normal')
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors='r', zorder=100, linewidth=1, label='Threshold')
ax.set_title('Reconstruction error for different classes')
ax.set_ylabel('Reconstruction error')
ax.set_xlabel('Data point index')
plt.legend()
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('images', 'fraud_detection_threshold.png'))
else:
    plt.show()

labels = ['Normal', 'Fraud']
pred_y = [1 if e > threshold_fixed else 0 for e in error_df.rec_error.values]
conf_matrix = confusion_matrix(error_df.true_class, pred_y)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.tight_layout()
if save_plots:
    plt.savefig(os.path.join('images', 'fraud_detection_conf_matrix.png'))
else:
    plt.show()