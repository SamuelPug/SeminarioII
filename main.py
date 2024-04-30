import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

C = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
F = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print('Entrenando')
historial = modelo.fit(C, F, epochs=1000, verbose=False)

print("Hagamos una predicción!")
# Crear un array de NumPy con el valor a predecir
valor_a_predecir = np.array([100.0], dtype=float)

# Hacer la predicción con el array de datos
resultado = modelo.predict(valor_a_predecir)
print("El resultado es " + str(resultado) + " fahrenheit!")