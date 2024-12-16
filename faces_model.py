import gdown
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Descargar el archivo zip desde Google Drive
folder_id = "1gMyyhzlsBeTekEvnuzfT1QQaysFBbUqp"  # Aquí coloca tu ID de la carpeta
url = f"https://drive.google.com/uc?export=download&id={folder_id}"
output = "/path/to/downloaded_folder"  # Ruta donde guardarás las imágenes
gdown.download(url, output, quiet=False)

# Una vez descargadas las imágenes, puedes usar este código para cargarlas
CARPETA_DATOS = output  # Ruta donde se descargaron las imágenes
TAMANO_IMG = 160  # Tamaño de las imágenes
AUTOTUNE = tf.data.AUTOTUNE  # Para optimizar el rendimiento

# Función para cargar y preprocesar imágenes desde Google Drive
def load_and_preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)  # Lee el archivo de la imagen
    img = tf.image.decode_jpeg(img, channels=3)  # Decodifica la imagen
    img = tf.image.resize(img, [TAMANO_IMG, TAMANO_IMG])  # Redimensiona la imagen
    img = img / 255.0  # Normaliza la imagen
    return img, label

# Función para obtener las rutas de las imágenes y sus etiquetas
def get_dataset(carpeta):
    imagenes = []
    labels = []
    for carpeta_persona in os.listdir(carpeta):
        path_carpeta_persona = os.path.join(carpeta, carpeta_persona)
        if os.path.isdir(path_carpeta_persona):
            for imagen in os.listdir(path_carpeta_persona):
                imagenes.append(os.path.join(path_carpeta_persona, imagen))
                labels.append(carpeta_persona)
    
    # Convertir las listas a un dataset de TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices((imagenes, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    return dataset

# Cargar los datos desde Google Drive
dataset = get_dataset(CARPETA_DATOS)

# Mostrar algunos datos
dataset_list = list(dataset)
print(f"Cantidad de datos cargados: {len(dataset_list)}")

# Dividir en datos de entrenamiento y prueba
imagenes, etiquetas = zip(*dataset_list)
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

X_train, X_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

# Visualizar algunas imágenes del conjunto de entrenamiento
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i])
    plt.title(f"Etiqueta: {y_train[i]}")
    plt.axis('off')
plt.show()

# Aquí puedes continuar con la definición y entrenamiento de tu modelo...
