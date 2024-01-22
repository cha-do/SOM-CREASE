# %% libraries
from minisom import MiniSom
import numpy as np
import pandas as pd
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
import multiprocessing as mp
from functools import partial

#%% Define the train_SOM function
def train_SOM(top, X, Y, lr, sg, epochs, data, pca=False, seed=None, verbose=False, address="./"):
    # Create and initialize the SOM
    som = MiniSom(X, Y, 7, sigma=sg, learning_rate=lr, topology=top, random_seed=seed)
    #Train the SOM using batches
    # batch_size = 10000
    # for j in range(num_epochs):
    #     np.random.shuffle(inSilicoData)
    #     sys.stdout.write("\r        Epoch {:d}/{:d}".format(j+1,num_epochs))
    #     sys.stdout.flush()
    #     for i in range(0, len(inSilicoData), batch_size):
    #         sys.stdout.write("\r{:d}".format(i+1))
    #         sys.stdout.flush()
    #         batch = inSilicoData[i:i + batch_size]
    #         som.train_random(batch, 1)
    # Train the SOM normaly
    if pca:
        som.pca_weights_init(data)
    else:
        som.random_weights_init(data)
    som.train_random(data, epochs)
    #som.train_batch(data_array, num_epochs, verbose=True)
    #Save the trained SOM
    namef = f'{top[:3]}_{X}x{Y}_lr{lr}_sg{sg}_eps{int(epochs/100)}E2_pca{int(pca)}_s{seed}'
    with open(f'{address}/SOM_{namef}.p', 'wb') as outfile:
        pickle.dump(som, outfile)
    # Evaluate the SOM
    somError = som.quantization_error(data[250000:280000])
    top_error = som.topographic_error(data[250000:280000])
    # Save the info
    with open(f'{address}/quality.txt','a') as f:
        f.write(f'{top[:3]} {X} {Y} {str(lr)} {str(sg)} {epochs} {int(pca)} {seed} ')
        f.write('%.4lf %.4lf\n'%(somError, top_error))
    # Visualice the distance_map of teh trained SOM
    if verbose:
        f = plt.figure(figsize=(10 ,10))
        ax = f.add_subplot(111)
        if top[:3] == "hex":
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
            ax.set_aspect('equal')
            # iteratively add hexagons
            plt.plot(0, 0)
            plt.plot(X-1, (Y-1)*np.sqrt(3)/2)
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)] * np.sqrt(3) / 2
                    hex = RegularPolygon((xx[(i, j)], wy), 
                                        numVertices=6, 
                                        radius=1 / np.sqrt(3),
                                        facecolor=cm.Blues(umatrix[i, j]))
                    ax.add_patch(hex)
            # plt.ylim(-1,Y)
            # plt.xlim(-1,X)
            xrange = np.arange(weights.shape[0])
            yrange = np.arange(weights.shape[1])
            plt.xticks(xrange-.5, xrange)
            plt.yticks(yrange * np.sqrt(3) / 2, yrange)

            divider = make_axes_locatable(plt.gca())
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
            cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                                        orientation='vertical', alpha=.4)
            cb1.ax.get_yaxis().labelpad = 100
            plt.gcf().add_axes(ax_cb)
        else:#top == "rec"
            plt.pcolor(som.distance_map().T, cmap=cm.Blues)  # Plot the distance_map
            plt.colorbar()
        plt.suptitle(namef)
        plt.savefig(address+'/'+namef+'_distanMap.png',dpi=169,bbox_inches='tight')

# %% 

#Trian multiple SOM's using mp
def train_SOMmp(i, works, address):
    print(f"Work {i}/{k+1}:")
    top = works[i]["top"]
    x = works[i]["X"]
    pca = works[i]["pca"]
    lr = works[i]["lr"]
    sg = works[i]["sg"]
    epochs = works[i]["epochs"]
    print(f'{top[:3]}_{x}x{x}_lr{lr}_sg{sg}_eps{int(epochs/100)}E2_pca{int(pca)}_s{seed}')
    train_SOM(top, x, x, lr, sg, epochs, inSilicoData, pca, seed, False, address)
    
# %%
#
if __name__ == "__main__":
    #%% Prepare works
    address = "./trainedSOMs"
    with open(f'{address}/quality.txt','a') as f:
        f.write("Topo X Y LearnRate Sigma Epochs PCA Seed quant_err topo_error\n")
    # Set the SOM parameters
    XY = [
        30,
        40,
        50
        ]
    lrs = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9
        ]
    sgs = [
        1.8,
        1.9,
        2.,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8
        ]
    n_epochs = [
        4500,
        5000,
        5500,
        6000,
        6500,
        7000,
        7500,
        8000,
        8500,
        9000,
        9500,
        10000,
        10500,
        11000,
        11500,
        12000
        ]
    seed = 10
    topologies = [
        'hexagonal',
        'rectangular'
        ]
    pcas = [
        True,
        False
        ]
    np.random.seed(seed)
    inSilicoData = np.random.randint(0, 128, size=(1000000, 7), dtype=int)/127
    n_cores = 8
    # %% set works
    works = {}
    k = 0
    for top in topologies:
        for x in XY:
            for pca in pcas:
                for lr in lrs:
                    for sg in sgs:
                        for epochs in n_epochs:
                            works[k] = {"top":top,
                                        "X":x,
                                        "pca":pca,
                                        "lr":lr,
                                        "sg":sg,
                                        "epochs":epochs}
                            k+=1
    print("Total Works:", k)
    firstwork = 0
    w = range(firstwork,k)#[0,1,2,3,4,5]
    pool = mp.Pool(n_cores)
    partial_work = partial(train_SOMmp,
                        works = works,
                        address = address)
    pool.map(partial_work,[i for i in w])
    pool.close()
    pool.join
