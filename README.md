# Tutorial Deep Learning con Tensorflow/Keras

## Instalación

### Con Anaconda

1. Instalar Anaconda:
   
    https://www.anaconda.com/
   
2. Importar el entorno (archivo ```envoronment.yml```)
        
        conda env create -f envoronment.yml

### Sin Anaconda

- [Instrucciones de instalación](https://www.tensorflow.org/install/pip?hl=es_419)
- [Instrucciones para usar la GPU](https://www.tensorflow.org/install/gpu?hl=es_419)

## Red totalmente conectada

### Entrenamiento del modelo

En este ejemplo utilizaremos el dataset ```CIFAR10``` para entrenar un clasificador

- Ejemplo del tipo de imágenes del dataset:

![Ejemplo CIFAR10](images/cifar10_tf_fcn_train.png)

Ejecuta el script ```cifar10_tf_fcn.py``` para entrenar el modelo

Este es un ejemplo de entrenamiento, en el que se ve que hay sobreentrenamiento (overfitting) a partir de la `epoch` 80 aproximadamente:

![Ejemplo CIFAR10](images/cifar10_tf_fcn_history.png)

Y esta imagen muestra un ejemplo visual de los resultados con el modelo anterior:

![Ejemplo CIFAR10](images/cifar10_tf_fcn_test.png)

En este caso, la precisión (accuracy) del modelo es:

      test accuracy: 0.5267

### Ajuste de hiperparámetros

Para mejorar el resultado se pueden modificar los siguientes parámetros:

- Número de capas
- Número de unidades en cada capa
- Optimizador: [opciones](https://keras.io/api/optimizers/)
- Función de coste: [opciones](https://keras.io/api/losses/)
- `learning_rate`: [explicación](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)
- `n_epochs`: [explicación](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
- `batch_size`: [explicación](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/#:~:text=The%20number%20of%20epochs%20is%20traditionally%20large%2C%20often%20hundreds%20or,500%2C%201000%2C%20and%20larger.)
- `validation_split`: [explicación](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
- `activation`: [opciones](https://keras.io/api/layers/activations/)


### Resultados

Podemos ir añadiendo los resultados con distintas configuraciones en el hilo de [discussions](https://github.com/esansano/tutorial-dl/discussions)

## Red convolucional

Ejecuta el script ```cifar10_tf_cnn.py``` para entrenar el modelo

### Resultados sin *Data Augmentation* (`data_augmentation = False`)

Este es un ejemplo de entrenamiento, en el que se ve que hay sobreentrenamiento (overfitting) a partir de la `epoch` 15~20 aproximadamente:

![Ejemplo CIFAR10](images/cifar10_tf_cnn_history_daf.png)


En este caso, la precisión (accuracy) del modelo es:

      test accuracy: 0.7184

### Resultados con *Data Augmentation* (`data_augmentation = True`)

Este es un ejemplo de entrenamiento en el que no hay *overfitting*:

![Ejemplo CIFAR10](images/cifar10_tf_cnn_history_dat.png)


En este caso, la precisión (accuracy) del modelo es:

      test accuracy: 0.7931

### Ajuste de hiperparámetros

Para mejorar el resultado se pueden modificar los siguientes parámetros:

- Activar/Desactivar `data_augmentation` y los parámetros que modifican las imágenes de entrenamiento
- Número de capas
- Número de unidades en cada capa
- Parámetros de las capas convolucionales (`kernel_size`, `strides`, `filters`): [explicación](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac)
- Capas `BatchNormalization`: [explicación](https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/)
- Dropout: [explicación](https://towardsdatascience.com/an-intuitive-explanation-to-dropout-749c7fb5395c)
- Optimizador: [opciones](https://keras.io/api/optimizers/)
- Función de coste: [opciones](https://keras.io/api/losses/)
- `learning_rate`: [explicación](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)
- `n_epochs`: [explicación](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
- `batch_size`: [explicación](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/#:~:text=The%20number%20of%20epochs%20is%20traditionally%20large%2C%20often%20hundreds%20or,500%2C%201000%2C%20and%20larger.)
- `validation_split`: [explicación](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
- `activation`: [opciones](https://keras.io/api/layers/activations/)

## Autoencoders

### Ejemplo 1: Detección de fraude

Ejecuta el script ```fraud_detection_tf_ae.py``` para entrenar el modelo

#### Aprendizaje no supervisado

Se entrena el modelo únicamente con los ejemplos de transacciones normales

![Ejemplo detección fraude](images/fraud_detection_tf_ae_history.png)


Una vez entrenado, utilizamos el conjunto de validación para buscar un umbral sobre el error de reconstrucción que nos permita separar las transacciones fraudulentas de las normales.

![Histograma detección fraude](images/fraud_detection_tf_ae_histogram.png)

Zoom:

![Histograma detección fraude](images/fraud_detection_tf_ae_histogram_zoom.png)

Se ve claramente que tenemos dos distribuciones distintas, aunque se solapan en algún intervalo.

Representamos algunas métricas (precision, recall y f1):

![precision-recall-f1 detección fraude](images/fraud_detection_tf_ae_prec_rec.png)

![precision-recall](images/precision_recall.jpg)

Como ejemplo, si elegimos un umbral para el error de reconstrucción de 5, obtendríamos esta separación de las clases:

![precision-recall](images/fraud_detection_tf_ae_threshold.png)

y esta matriz de confución:

![precision-recall](images/fraud_detection_tf_ae_conf_matrix.png)

### Ejemplo 2: Mejora de la resolución de imágenes

En este ejemplo utilizamos las imágenes en baja resolución como entrada al `autoencoder` y las imágenes con alta resolución como salida.

Este es un ejemplo del tipo de imágenes utilizadas. A la izquierda la entrada y a la derecha la salida.

![Image enhance train set](images/img_enhance_tf_ae_train.png)

Entrenamos hasta que el coste en el conjunto de validación deja de bajar:

![Image enhance train history](images/img_enhance_tf_ae_history.png)

Este es el resultado evaluando en el conjunto de test:

![Image enhance test set evaluation](images/img_enhance_tf_ae_test.png)

## NLP (Natural Language Processing)

### Word embedding

[Skip-gram embedding](https://gruizdevilla.medium.com/introducci%C3%B3n-a-word2vec-skip-gram-model-4800f72c871f)

![word embedding process](images/word_embedding_skip_gram.png)

![word embedding vectors](images/word_embedding_vectors.png)


- [king] - [man] + [woman] -> [queen]
- [paris] - [france] + [italy] -> [rome]


### Transformer

#### Attention

[Attention mechanism](https://medium.com/analytics-vidhya/https-medium-com-understanding-attention-mechanism-natural-language-processing-9744ab6aed6a)

![Mecanismo de atención](images/attention_mechanism.png)

#### Transformer completo

[Transformer](http://jalammar.github.io/illustrated-transformer/)

![Transformer original](images/complete_transformer.png)


### Ejemplos

#### Ejemplo 1

Análisis de sentimiento. El conjunto de entrenamiento está compuesto por críticas de cine etiquetadas como positivas o negativas.

El modelo está compuesto por una capa de *embedding*, dos capas convolucionales de 1 sola dimensión, y un clasificador totalmente conectado.


script: `nlp_text_class_tf.py`

#### Ejemplo 2

Este ejemplo es igual que el anterior, pero sustituyendo las dos capas convolucionales por una versión reducida de *transformer* que solo incluye la parte del *encoder*

script: `nlp_text_class_transformer_tf.py`

En ambos casos, el resultado en el conjunto de test está alrededor de 0.84 (accuracy)