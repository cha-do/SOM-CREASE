from minisom import MiniSom
import numpy as np

# Cargar tus datos (reemplaza 'datos.npy' con tu propio archivo de datos)
datos = np.random.rand(1000, 7)

# Normalizar los datos
datos_norm = (datos - np.min(datos, axis=0)) / (np.max(datos, axis=0) - np.min(datos, axis=0))

# Tamaño de la cuadrícula hexagonal
tamanio_mapa = (10, 10)  # Ajusta según tus necesidades

# Crear e inicializar el SOM
som = MiniSom(tamanio_mapa[0], tamanio_mapa[1], datos_norm.shape[1], sigma=0.3, learning_rate=0.5)
som.random_weights_init(datos_norm)

# Entrenar el SOM
num_epocas = 100
som.train_random(datos_norm, num_epocas)

# Visualizar el mapa de distancias de las neuronas
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot del mapa de distancias
plt.colorbar()

# Plotear los datos en el SOM
for i, x in enumerate(datos_norm):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', markersize=10, markeredgecolor='red', markeredgewidth=2)

plt.show()
