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

[Attention mechanism (1)](https://medium.com/analytics-vidhya/https-medium-com-understanding-attention-mechanism-natural-language-processing-9744ab6aed6a)

[Attention mechanism (2)](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853)

![Mecanismo de atención](images/attention_mechanism.png)

#### Transformer completo

[Transformer](http://jalammar.github.io/illustrated-transformer/)

![Transformer original](images/complete_transformer.png)


### Ejemplos

#### Ejemplo 1

Análisis de sentimiento. El conjunto de entrenamiento está compuesto por críticas de cine etiquetadas como positivas o negativas.

El modelo está compuesto por una capa de *embedding*, dos capas convolucionales de 1 sola dimensión, y un clasificador totalmente conectado.


script: `nlp_text_class_tf.py`

[Pooling Layers](https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/)

#### Ejemplo 2

Este ejemplo es igual que el anterior, pero sustituyendo las dos capas convolucionales por una versión reducida de *transformer* que solo incluye la parte del *encoder*

script: `nlp_text_class_transformer_tf.py`

En ambos casos, el resultado en el conjunto de test está alrededor de 0.84 (accuracy)

## Recurrent Neural Networks

[RNN - LSTM](https://medium.com/@humble_bee/rnn-recurrent-neural-networks-lstm-842ba7205bbf)

[Vanilla RNN](https://datascience-enthusiast.com/DL/Building_a_Recurrent_Neural_Network-Step_by_Step_v1.html)

[LSTM](https://adventuresinmachinelearning.com/keras-lstm-tutorial/)

[Tutorial RNN](https://www.cs.toronto.edu/~tingwuwang/rnn_tutorial.pdf)

#### Ejemplo 1

Este ejemplo es igual que los dos anteriores, pero sustituyendo las capas convolucionales/transformer por una capa de celdas LSTM.

script: `nlp_text_class_lstm_tf.py`

El resultado en el conjunto de test es parecido a los ejemplos anteriores, pero el entrenamiento es más lento.


## Casos de uso, arquitecturas y ejemplos

### Computer vision

[Arquitecturas para *computer vision*](https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/)


[Applications of GANs](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/)

[GAN example with Keras](https://keras.io/examples/generative/dcgan_overriding_train_step/)

### Speech recognition

[Speech recognition](https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706)

### Natural Language Processing

[Applications of deep learning in NLP](https://medium.com/@datamonsters/artificial-neural-networks-in-natural-language-processing-bcf62aa9151a)

### Recommendation engines

[Deep learning-based recommender systems (1)](https://medium.com/sciforce/deep-learning-based-recommender-systems-b61a5ddd5456)


[Deep learning-based recommender systems (2)](https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e)


### Health

[3D Image Classification from CT Scans (pneumonia prediction)](https://keras.io/examples/vision/3D_image_classification/)

[Anomaly detection in wearable health data](https://towardsdatascience.com/does-machine-learning-know-when-im-scared-anomaly-detection-in-wearable-health-data-72f537dd4370)


## Recursos

[Curso gratuito](https://d2l.ai/index.html)