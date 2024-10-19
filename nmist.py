import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import time

# Cargar el dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# Normalizar los datos
x_train, x_test = x_train / 255.0, x_test / 255.0

#print(f'Tamaño del conjunto de entrenamiento: {x_train.shape}')

x_train = x_train.reshape(60000, 28*28) 
x_test = x_test.reshape(10000, 28*28)


# Definir el modelo
model = models.Sequential([
    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.Flatten(),
    # layers.Dense(64, activation='relu'),
    # layers.Dense(10, activation='softmax') 

	layers.Dense(5, input_dim=28*28, activation='sigmoid'),
    # for c in range(capas):
    #layers.Dense(5, activation='sigmoid'),
    layers.Dense(10,activation='softmax')
])

# Compilar el modelo
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer=optimizers.SGD(), loss='mean_squared_error')

start=time.time()
# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Tiempo de entrenamiento: ",time.time()-start)


# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Precisión en el conjunto de prueba: {test_acc}')

import numpy as np
# Calcular el error cuadrático medio manualmente
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # Obtener las etiquetas predichas

# Usar mean_squared_error para calcular la pérdida manualmente
mse_loss = np.mean(tf.keras.losses.mean_squared_error(y_test, y_pred_labels).numpy())
print(f'MSE calculado manualmente: {mse_loss}')