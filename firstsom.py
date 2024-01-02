import somoclu
import numpy as np

# Genera datos de ejemplo (ajusta según tus datos)
np.random.seed(42)
data = np.random.rand(1000, 7)

# Configura y entrena el SOM paralelo
n_rows, n_columns = 10, 10
som = somoclu.Somoclu(n_rows, n_columns, compactsupport=False)
som.train(data)

# Visualiza el mapa de distancias de las neuronas
som.view_umatrix(bestmatches=True, colorbar=True, figsize=(10, 8))
