from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(1000, 7)  # Ejemplo de datos aleatorios (reemplázalo con tus datos reales)
y = np.random.randint(0, 2, size=(1000,))  # Ejemplo de etiquetas aleatorias (0 o 1)

# Paso 1: Preprocesamiento de datos
# Supongamos que X es tu conjunto de datos de 7 dimensiones y y es la variable de salida continua
# Normaliza los datos si es necesario

# Paso 2: Configuración del SOM
som_size = (10, 10)  # Tamaño del SOM
input_dim = 7  # Número de dimensiones de entrada
som = MiniSom(som_size[0], som_size[1], input_dim, sigma=1.0, learning_rate=0.5)

# Paso 3: Entrenamiento del SOM
num_iterations = 1000
som.train_random(X, num_iterations, verbose=True)

# Paso 4: Visualización del Mapa 2D de Calor
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Plot del mapa de distancias
plt.colorbar()

for i, x in enumerate(X):
    winner = som.winner(x)  # Encuentra la neurona ganadora para el dato x
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markerfacecolor=plt.cm.RdYlBu(y[i]/np.max(y)), markersize=12, markeredgewidth=2)

plt.show()
