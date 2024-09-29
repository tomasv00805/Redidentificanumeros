from __future__ import absolute_import, division, print_function, unicode_literals

# Importa las bibliotecas necesarias
import tensorflow as tf
import tensorflow_datasets as tfds  # Biblioteca para cargar datasets, en este caso MNIST

import math  # Biblioteca matemática para operaciones de redondeo
import numpy as np  # Biblioteca para manejo de arrays y operaciones numéricas
import matplotlib.pyplot as plt  # Biblioteca para graficar los resultados
import logging  # Biblioteca para manejo de logs

from urllib import parse  # Biblioteca para manejo de URLs, usada en la recepción de datos
from http.server import HTTPServer, BaseHTTPRequestHandler  # Biblioteca para crear un servidor HTTP

# Configuración de logging para que solo muestre errores de TensorFlow
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Carga del dataset de MNIST, cargando los sets de entrenamiento y prueba
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Obtención del número de ejemplos en los datasets de entrenamiento y prueba
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# Normalización de los datos de imagen: de rango 0-255 a rango 0-1 para mejorar la eficiencia del modelo
def normalize(images, labels):
    images = tf.cast(images, tf.float32)  # Convierte los valores a float32
    images /= 255  # Divide por 255 para normalizar
    return images, labels

# Aplicamos la normalización a los datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Estructura de la red neuronal
model = tf.keras.Sequential([  # Modelo secuencial
    tf.keras.layers.Flatten(input_shape=(28,28,1)),  # Aplana las imágenes de 28x28 a un vector de 784 elementos
    tf.keras.layers.Dense(124, activation=tf.nn.relu),  # Capa densa (fully connected) con 124 neuronas y activación ReLU
    tf.keras.layers.Dense(64, activation=tf.nn.relu),  # Segunda capa densa con 64 neuronas y activación ReLU
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Capa de salida con 10 neuronas (una para cada dígito), activación softmax para clasificación
])

# Compilación del modelo especificando el optimizador, la función de pérdida y la métrica de precisión
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación con etiquetas enteras
    metrics=['accuracy']  # Métrica de evaluación (precisión)
)

# Preparación del entrenamiento por lotes
BATCHSIZE = 32  # Tamaño del lote
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)  # Repetir, barajar y agrupar en lotes
test_dataset = test_dataset.batch(BATCHSIZE)  # Agrupar en lotes el dataset de prueba

# Entrenamiento del modelo durante 5 épocas
history = model.fit(
    train_dataset, epochs=5,
    steps_per_epoch=math.ceil(num_train_examples/BATCHSIZE)  # Número de pasos por época
)

# Graficar la historia de entrenamiento
# Extraemos los datos de historia del entrenamiento
acc = history.history['accuracy']  # Precisión en cada época
loss = history.history['loss']  # Pérdida en cada época
epochs_range = range(5)  # Rango de épocas

# Crear las gráficas de precisión y pérdida
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)  # Primera gráfica
plt.plot(epochs_range, acc, label='Precisión')
plt.legend(loc='lower right')
plt.title('Precisión durante el entrenamiento')

plt.subplot(1, 2, 2)  # Segunda gráfica
plt.plot(epochs_range, loss, label='Pérdida')
plt.legend(loc='upper right')
plt.title('Pérdida durante el entrenamiento')
plt.show()

# Clase para manejar solicitudes HTTP POST, que recibe una imagen en formato de cadena
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Petición recibida")  # Log para indicar que se ha recibido una petición

        # Obtiene la longitud de los datos en la petición y los lee
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        
        # Decodifica los datos y elimina el prefijo "pixeles="
        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)  # Desescapa los caracteres especiales

        # Convierte los datos de la petición en un array y lo transforma para que coincida con las imágenes de MNIST
        arr = np.fromstring(data, np.float32, sep=",")  # Convierte la cadena a un array NumPy
        arr = arr.reshape(28,28)  # Lo reorganiza en una imagen de 28x28
        arr = np.array(arr)
        arr = arr.reshape(1,28,28,1)  # Le añade una dimensión para que tenga el mismo formato que las imágenes de MNIST

        # Realiza la predicción con el modelo entrenado
        prediction_values = model.predict(arr, batch_size=1)
        prediction = str(np.argmax(prediction_values))  # Obtiene el dígito con mayor probabilidad
        print("Predicción final: " + prediction)

        # Responde a la petición HTTP con la predicción
        self.send_response(200)
        # Evita problemas de CORS permitiendo peticiones desde cualquier origen
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(prediction.encode())  # Envía la predicción como respuesta

# Inicia un servidor HTTP en localhost en el puerto 8000
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()  # Mantiene el servidor en ejecución indefinidamente
