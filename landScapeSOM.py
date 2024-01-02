# %% libraries
from minisom import MiniSom
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt

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
    normaldataunique[column] = (dataunique[column] - min_val) / (max_val - min_val)

# Display the normalized DataFrame
print(normaldataunique)
#%%
# Tamaño de la cuadrícula hexagonal
tamanio_mapa = (20, 20)  # Ajusta según tus necesidades

# Crear e inicializar el SOM
som = MiniSom(tamanio_mapa[0], tamanio_mapa[1], 7, sigma=0.2, learning_rate=0.3)
data_array = normaldataunique.values
som.random_weights_init(data_array)
# Entrenar el SOM
num_epochs = 10000
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
som.train_batch(data_array, num_epochs, verbose=True)
#Save the trained SOM
with open(f'som{tamanio_mapa[0]}x{tamanio_mapa[1]}.p', 'wb') as outfile:
    pickle.dump(som, outfile)
# %% load a trained SOM
with open('som.p', 'rb') as infile:
    som_trained = pickle.load(infile)
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

alldata.to_csv("neurons_"+csv_files_path[k], index=False)
# %% Load neurons
alldata = pd.read_csv("neurons_"+csv_files_path[k])
#%%
sseavg = np.ones(tamanio_mapa)
ssemin = np.ones(tamanio_mapa)
ssemax = np.ones(tamanio_mapa)
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
plt.pcolor(sseavg, cmap='bone_r')  # Plot del mapa de distancias
plt.colorbar()
plt.show()

# %%
