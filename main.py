import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
#comentario1

base_dir = "c:/Users/marce/OneDrive/Escritorio/PruebaFactibilidad/gestoss"
gestos = os.listdir("gestoss") # Lista de carpetas con los nombres de los gestos.
X = []  # Para almacenar las imágenes.
y = []  # Para almacenar las etiquetas.

for idx, gesto in enumerate(gestos):
    print(f"Procesando {gesto}...")
    for imagen_nombre in os.listdir(os.path.join(base_dir, gesto)):
        imagen_path = os.path.join(base_dir, gesto, imagen_nombre)
        
        # Usamos PIL para abrir y procesar las imágenes.
        imagen = Image.open(imagen_path)
        imagen = imagen.resize((128, 128))  # Redimensionar.
        imagen = np.array(imagen) / 255.0  # Normalizar.
        
        X.append(imagen)
        y.append(idx)

X = np.array(X)
y = to_categorical(y, num_classes=len(gestos))

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.5,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% entrenamiento, 30% temporal.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validación, 15% prueba.

plt.imshow(X_train[0])
plt.title(np.argmax(y_train[0]))
plt.show()


import tensorflow as tf

def create_cnn(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

cnn_model = create_cnn((128, 128, 3), len(gestos))


cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 50
batch_size = 32

history = cnn_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs
)

test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")

import matplotlib.pyplot as plt

# Gráfica de precisión
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Gráfica de pérdida
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

cnn_model.save("MiNeurona.h5")
# Importar el convertidor
import tensorflow as tf

# Cargar el modelo HDF5
loaded_model = tf.keras.models.load_model("MiNeurona.h5")

# Convertir el modelo
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Guardar el modelo convertido
with open("MiNeurona.tflite", "wb") as f:
    f.write(tflite_model)
