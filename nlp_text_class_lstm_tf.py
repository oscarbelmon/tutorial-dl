import os
import re
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.data.experimental import cardinality
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, LSTM
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
raw_test_ds = text_dataset_from_directory(os.path.join(data_path, 'test'), batch_size=batch_size)

print(f'Number of batches in raw_train_ds: {cardinality(raw_train_ds)}')
print(f'Number of batches in raw_val_ds: {cardinality(raw_val_ds)}')
print(f'Number of batches in raw_test_ds: {cardinality(raw_test_ds)}')

# Imprimimos 5 instancias para ver que aspecto tienen los datos
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


# Esta función se encargará de limpiar los datos (quitar las etiquetas '<br />' y los escapes
# antes de los signos de puntuación) y pasar el texto a minúsculas.
def clean_input(input_data):
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
epochs = 15
learning_rate = 0.001

# Esto convierte una frase en un vector de enteros. Cada número representa el índice de la palabra
# en el vocabulario
vectorize_layer = TextVectorization(standardize=clean_input, max_tokens=vocabulary_size, output_mode='int',
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
x = LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)(x)
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

# output
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, None)]            0         
_________________________________________________________________
embedding (Embedding)        (None, None, 32)          640000    
_________________________________________________________________
dropout (Dropout)            (None, None, 32)          0         
_________________________________________________________________
lstm (LSTM)                  (None, 128)               82432     
_________________________________________________________________
dense (Dense)                (None, 128)               16512     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
predictions (Dense)          (None, 1)                 129       
=================================================================
Total params: 739,073
Trainable params: 739,073
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/15
625/625 [==============================] - 66s 102ms/step - loss: 0.6934 - accuracy: 0.5016 - val_loss: 0.6928 - val_accuracy: 0.5084
Epoch 2/15
625/625 [==============================] - 66s 106ms/step - loss: 0.6884 - accuracy: 0.5342 - val_loss: 0.6780 - val_accuracy: 0.5520
Epoch 3/15
625/625 [==============================] - 66s 106ms/step - loss: 0.6421 - accuracy: 0.6074 - val_loss: 0.7018 - val_accuracy: 0.5536
Epoch 4/15
625/625 [==============================] - 66s 106ms/step - loss: 0.5763 - accuracy: 0.7012 - val_loss: 0.6635 - val_accuracy: 0.6026
Epoch 5/15
625/625 [==============================] - 66s 106ms/step - loss: 0.6472 - accuracy: 0.6118 - val_loss: 0.6930 - val_accuracy: 0.5550
Epoch 6/15
625/625 [==============================] - 66s 106ms/step - loss: 0.6285 - accuracy: 0.6169 - val_loss: 0.6845 - val_accuracy: 0.5374
Epoch 7/15
625/625 [==============================] - 66s 106ms/step - loss: 0.6422 - accuracy: 0.6022 - val_loss: 0.6815 - val_accuracy: 0.6626
Epoch 8/15
625/625 [==============================] - 66s 106ms/step - loss: 0.4272 - accuracy: 0.8150 - val_loss: 0.4067 - val_accuracy: 0.8354
Epoch 9/15
625/625 [==============================] - 66s 106ms/step - loss: 0.2675 - accuracy: 0.8979 - val_loss: 0.3663 - val_accuracy: 0.8520
Epoch 10/15
625/625 [==============================] - 66s 106ms/step - loss: 0.1952 - accuracy: 0.9299 - val_loss: 0.3918 - val_accuracy: 0.8630
Epoch 11/15
625/625 [==============================] - 66s 106ms/step - loss: 0.1596 - accuracy: 0.9436 - val_loss: 0.4381 - val_accuracy: 0.8414
Epoch 12/15
625/625 [==============================] - 66s 106ms/step - loss: 0.1272 - accuracy: 0.9553 - val_loss: 0.4212 - val_accuracy: 0.8458
Epoch 13/15
625/625 [==============================] - 66s 106ms/step - loss: 0.1111 - accuracy: 0.9611 - val_loss: 0.4640 - val_accuracy: 0.8462
Epoch 14/15
625/625 [==============================] - 66s 106ms/step - loss: 0.0901 - accuracy: 0.9694 - val_loss: 0.4972 - val_accuracy: 0.8490
Epoch 15/15
625/625 [==============================] - 66s 106ms/step - loss: 0.0783 - accuracy: 0.9721 - val_loss: 0.4553 - val_accuracy: 0.8630
Test accuracy
782/782 [==============================] - 11s 14ms/step - loss: 0.5874 - accuracy: 0.8333

"""
