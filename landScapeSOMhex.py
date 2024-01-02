# %% libraries
from minisom import MiniSom
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Ellipse
from matplotlib import cm, colorbar

# %% load data
# Specify the path to your CSV file
csv_files_path = [
    "1_10_12_6_12__AGA.csv",
    "2_10_6_12_6__AGA.csv",
    "3_15_12_6_12__AGA.csv",
    "4_15_6_12_6__AGA.csv",
    ]
k = 1
# Read the CSV file into a DataFrame
alldata = pd.read_csv(csv_files_path[k])
# Display the DataFrame
print(alldata)
# %% drop duplicates
paramsname = ["seed","R_core", "t_Ain", "t_B", "t_Aout", "sAin", "sigmaR", "-log(bg)", "SSE"]
dataunique = alldata[paramsname[1:8]].drop_duplicates()

print(dataunique)
#%% Normalize data
# Define custom minimum and maximum values for each column
minvalu = np.array([50, 30, 30, 30, 0.1, 0.0, 2.5])
maxvalu = np.array([250, 200, 200, 200, 0.45, 0.45, 5.5])

# Normalize each column to the specified custom range
normaldataunique = pd.DataFrame()

for i in range(len(minvalu)):
    min_val = minvalu[i]
    max_val = maxvalu[i]
    column = paramsname[i+1]
    normaldataunique[column] = ((dataunique[column] - min_val) / (max_val - min_val))
# normaldataunique = (dataunique - np.mean(dataunique, axis=0)) / np.std(dataunique, axis=0)
# Display the normalized DataFrame
print(normaldataunique)
#%%
# Tamaño de la cuadrícula hexagonal
tamanio_mapa = (15, 15)  # Ajusta según tus necesidades
array_resultante = np.random.randint(0, 128, size=(250000, 7), dtype=int)/128
# Crear e inicializar el SOM
#som = MiniSom(tamanio_mapa[0], tamanio_mapa[1], 7, sigma=0.2, learning_rate=0.3)
som = MiniSom(tamanio_mapa[0], tamanio_mapa[1], 7, sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              neighborhood_function='gaussian', random_seed=10)
data_array = normaldataunique.values
#np.random.shuffle(data_array)
#som.random_weights_init(data_array)
# Entrenar el SOM
num_epochs = 50000
# batch_size = 10000
# for j in range(num_epochs):
#     np.random.shuffle(data_array)
#     sys.stdout.write("\r        Epoch {:d}/{:d}".format(j+1,num_epochs))
#     sys.stdout.flush()
#     for i in range(0, len(data_array), batch_size):
#         sys.stdout.write("\r{:d}".format(i+1))
#         sys.stdout.flush()
#         batch = data_array[i:i + batch_size]
#         som.train_random(batch, 1)
#som.train(data_array, num_epochs, verbose=True)
# som.pca_weights_init(array_resultante)
som.train_random(array_resultante, num_epochs, verbose=True)
#som.train_batch(data_array, num_epochs, verbose=True)
#Save the trained SOM
with open(f'som{tamanio_mapa[0]}x{tamanio_mapa[1]}.p', 'wb') as outfile:
    pickle.dump(som, outfile)
# %% load a trained SOM
tamanio_mapa = (15, 15)
with open(f'som{tamanio_mapa[0]}x{tamanio_mapa[1]}.p', 'rb') as infile:
    som_trained = pickle.load(infile)
#%%
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

f = plt.figure(figsize=(10,10))
ax = f.add_subplot(111)

ax.set_aspect('equal')

# iteratively add hexagons
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        wy = yy[(i, j)] * np.sqrt(3) / 2
        hex = RegularPolygon((xx[(i, j)], wy), 
                             numVertices=6, 
                             radius=.95 / np.sqrt(3),
                             facecolor=cm.Blues(umatrix[i, j]), 
                             alpha=.4, 
                             edgecolor='gray')
        ax.add_patch(hex)
plt.show()
#%%
markers = ['o', '+', 'x']
colors = ['C0', 'C1', 'C2']
for cnt, x in enumerate(data):
    # getting the winner
    w = som.winner(x)
    # place a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w) 
    wy = wy * np.sqrt(3) / 2
    plt.plot(wx, wy, 
             markers[t[cnt]-1], 
             markerfacecolor='None',
             markeredgecolor=colors[t[cnt]-1], 
             markersize=12, 
             markeredgewidth=2)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange * np.sqrt(3) / 2, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

legend_elements = [Line2D([0], [0], marker='o', color='C0', label='Kama',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='+', color='C1', label='Rosa',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='x', color='C2', label='Canadian',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left', 
          borderaxespad=0., ncol=3, fontsize=14)

plt.savefig('resulting_images/som_seed_hex.png')
plt.show()
#%%
# Get the winning neurons for each data point
# winning_neurons = som.winner(data_array)
indexUnique = normaldataunique.index.tolist()
data_array = normaldataunique.values
neurons = np.ones((alldata.shape[0],2))*-1
for i in range(len(indexUnique)):
    winning_neuron = som.winner(data_array[i])
    # Create a boolean mask for rows that are exactly equal to the chosen row
    row_to_match = dataunique.loc[indexUnique[i]]
    mask = (alldata['R_core'] == row_to_match['R_core']) & \
    (alldata['t_Ain'] == row_to_match['t_Ain']) & \
    (alldata['t_B'] == row_to_match['t_B']) & \
    (alldata['t_Aout'] == row_to_match['t_Aout']) & \
    (alldata['sAin'] == row_to_match['sAin']) & \
    (alldata['sigmaR'] == row_to_match['sigmaR']) & \
    (alldata['-log(bg)'] == row_to_match['-log(bg)'])
    indices = alldata[mask].index.tolist()
    neurons[indices] = np.array(winning_neuron)
alldata['Neuron_X'] = neurons[:,0]
alldata['Neuron_Y'] = neurons[:,1]
print(alldata)

alldata.to_csv(f'neurons_{tamanio_mapa[0]}x{tamanio_mapa[1]}_{csv_files_path[k]}', index=False)
# %% Load neurons
alldata = pd.read_csv(f'neurons_{tamanio_mapa[0]}x{tamanio_mapa[1]}_{csv_files_path[k]}')
#%%
sseavg = np.ones(tamanio_mapa)*100
ssemin = np.ones(tamanio_mapa)*100
ssemax = np.ones(tamanio_mapa)*100
for i in range(tamanio_mapa[0]):
    for j in range(tamanio_mapa[1]):
        indexes = alldata[(alldata['Neuron_X']==i) & (alldata['Neuron_Y']==j)].index.tolist()
        if indexes != []:
            sseavg[i,j] = alldata.loc[indexes]["SSE"].mean()
            ssemin[i,j] = alldata.loc[indexes]["SSE"].min()
            ssemax[i,j] = alldata.loc[indexes]["SSE"].max()

#%%
# Visualizar el mapa de distancias de las neuronas
sseavglog = np.log(sseavg)
plt.figure(figsize=(10, 10))
plt.pcolor(sseavglog, cmap='bone_r')  # Plot del mapa de distancias
plt.colorbar()
plt.show()

# %%
