import gdown
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

# Descargar imágenes desde Google Drive usando gdown
def download_images_from_drive():
    # Asegúrate de usar la URL de "export=download" de Google Drive
    urls = [
        'https://drive.google.com/uc?export=download&id=1gMyyhzlsBeTekEvnuzfT1QQaysFBbUqp'  # Enlace para una carpeta o archivo específico
    ]

    for url in urls:
        output = url.split('=')[-1]  # Extraer ID del archivo o nombre para guardarlo
        gdown.download(url, output, quiet=False)

# Llamamos a la función para descargar las imágenes
download_images_from_drive()

# Asegúrate de que las imágenes se hayan descargado y estén disponibles
dataset_directory = '/home/sakura/quinto/design'  # Cambia esta ruta según donde se guarden las imágenes

# Cargar el dataset
image_size = (150, 150)
batch_size = 32

# Aquí usamos ImageDataGenerator para cargar las imágenes de la carpeta descargada
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    dataset_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # Cambiar si es más de dos clases
)

# Dividir en conjunto de entrenamiento y prueba
# Dataset: Imágenes de entrenamiento
X_train, X_val, y_train, y_val = train_test_split(
    train_generator, test_size=0.2, random_state=42
)

# Aquí crearías tu modelo (por ejemplo, usando una red neuronal de Keras o TensorFlow)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Usar 'softmax' si hay más de 2 clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=(X_val, y_val)
)

# Ver los resultados
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.show()

# Puedes guardar el modelo entrenado
model.save('mi_modelo_entrenado.h5')
