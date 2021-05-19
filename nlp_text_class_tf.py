import os
import re
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.data.experimental import cardinality
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.python.keras.preprocessing.text_dataset import text_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.utils.vis_utils import plot_model


# descargar el dataset desde: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# una vez descomprimido, borrar la carpeta 'unsup' (sólo útil para aprendizaje no supervisado)
data_path = os.path.join('data', 'nlp', 'aclImdb')

batch_size = 32
raw_train_ds = text_dataset_from_directory(os.path.join(data_path, 'train'), batch_size=batch_size,
                                           validation_split=0.2, subset='training', seed=42)
raw_val_ds = text_dataset_from_directory(os.path.join(data_path, 'train'), batch_size=batch_size,
                                         validation_split=0.2, subset='validation', seed=42)
raw_test_ds = text_dataset_from_directory(os.path.join(data_path, 'train'), batch_size=batch_size)

print(f'Number of batches in raw_train_ds: {cardinality(raw_train_ds)}')
print(f'Number of batches in raw_val_ds: {cardinality(raw_val_ds)}')
print(f'Number of batches in raw_test_ds: {cardinality(raw_test_ds)}')

# Imprimimos 5 instancias para ver que aspecto tienen los datos
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


# Esta función se encargará de limpiar los datos (quitar las etiquetas '<br />' y los escapes
# antes de los signos de puntuación) y pasar toso el texto a minúsculas.
def custom_standardization(input_data):
    result = tf.strings.lower(input_data)
    result = tf.strings.regex_replace(result, '<br />', ' ')
    result = tf.strings.regex_replace(result, f'[{re.escape(string.punctuation)}]', '')
    return result


# Hiperparámetros
vocabulary_size = 20000
embedding_dim = 32
sequence_length = 200
dropout = 0.5
n_filters = 128
kernel_size = 7
epochs = 5
learning_rate = 0.001

# Esta capa convierte una frase en un vector de enteros. Cada número representa el índice de la palabra
# en el vocabulario
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=vocabulary_size, output_mode='int',
                                    output_sequence_length=sequence_length)

# text_ds es un data set que contiene solamente texto, no las etiquetas. Lo utilizamos
# para crear el vocabulario. La función adapt crea el vocabulario, con un número de palabras
# máximo de "vocabulary_size"
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


# definimos una función para vectorizar el data set completo. Esta función se encarga de recibir
# texto y devuelve un vector de enteros para cada frase. Los enteros son el índice de cada palabra en el
# vocabulario creado anteriormente
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Ejemplo de salida de la función anterior "vectorize_text"
example_text = 'The concept of the legal gray area in Love Crimes'
example_text_vectorized = vectorize_text([example_text], 0)[0]
print(example_text, example_text_vectorized)

# Usamos la función "vectoriza_text" para convertir el texto a números enteros
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Construimos el modelo para aprender a clasificar el sentimiento (Positivo - Negativo)
inputs = Input(shape=(None,), dtype='int64')
x = Embedding(vocabulary_size, embedding_dim)(inputs)
x = Dropout(rate=dropout)(x)
x = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='valid', activation='relu', strides=3)(x)
x = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='valid', activation='relu', strides=3)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(units=128, activation='relu')(x)
x = Dropout(rate=dropout)(x)
output = Dense(1, activation='sigmoid', name='predictions')(x)
model = Model(inputs, output)

print(model.summary())
plot_model(model, to_file=os.path.join('images', 'nlp_text_class_tf.png'), show_shapes=True, expand_nested=True)

# Optimizer (https://keras.io/api/optimizers/)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Loss (https://keras.io/api/losses/)
loss = keras.losses.BinaryCrossentropy()

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=epochs)
print('Test accuracy')
model.evaluate(test_ds)

# 782/782 [==============================] - 9s 11ms/step - loss: 0.1103 - accuracy: 0.9656
