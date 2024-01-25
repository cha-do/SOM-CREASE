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
from os import path
import multiprocessing as mp
from functools import partial

def use_SOM(k, works, paramsInterest=None, log = True):
    profile = works[k]["profile"]
    SOMparameters = works[k]["SOMparameters"]
    print(f'Work {k}: {nameSOM}')
    # %%
    top = SOMparameters[0]
    X = SOMparameters[1]
    Y = X
    lr = SOMparameters[2]
    sg = SOMparameters[3]
    epochs = SOMparameters[4]
    pca = SOMparameters[5]
    seed = SOMparameters[6]
    nameSOM = f'{top[:3]}_{X}x{Y}_lr{lr}_sg{sg}_eps{int(epochs/100)}E2_pca{pca}_s{seed}'
    # Read the CSV file into a DataFrame
    alldata = pd.read_csv(f"./Datasets/{profile}__AGA.csv")
    # Display the DataFrame
    # print(alldata)
    # %% drop duplicates
    paramsname = ["seed","R_core", "t_Ain", "t_B", "t_Aout", "sAin", "sigmaR", "-log(bg)", "SSE"]
    dataunique = alldata[paramsname[1:8]].drop_duplicates()
    # print(dataunique)
    #%% Normalize data
    # Normalize each column to the specified custom range
    normaldataunique = pd.DataFrame()
    for i in range(7):
        column = paramsname[i+1]
        normaldataunique[column] = (dataunique[column] - dataunique[column].min()) / (dataunique[column].max() - dataunique[column].min())
    # Display the normalized DataFrame
    # print(normaldataunique)
    # %% Load neurons
    if path.isfile(f"{profile}/neurons_{nameSOM}.csv"):
        alldata = pd.read_csv(f"{profile}/neurons_{nameSOM}.csv")
    else:
         # %% get the trained SOM
        SOMaddress = "./trainedSOMs"
        if path.is_file(f'{SOMaddress}/SOM_{nameSOM}.p'):
            #load the SOM
            with open(f'{SOMaddress}/SOM_{nameSOM}.p', 'rb') as infile:
                som = pickle.load(infile)
        else:
            # Create and initialize the SOM
            som = MiniSom(X, Y, 7, sigma=sg, learning_rate=lr, topology=top, random_seed=seed)
            if seed is not None:
                np.random.seed(seed)
            data = np.random.randint(0, 128, size=(10000000, 7), dtype=int)/127
            if pca:
                som.pca_weights_init(data)
            else:
                som.random_weights_init(data)
            som.train_random(data, epochs)
            #Save the trained SOM
            with open(f'{SOMaddress}/SOM_{nameSOM}.p', 'wb') as outfile:
                pickle.dump(som, outfile)
            # Evaluate the SOM
            somError = som.quantization_error(data[750000:780000])
            top_error = som.topographic_error(data[750000:780000])
            # Save the info
            with open(f'{SOMaddress}/quality.txt','a') as f:
                f.write(f'{top[:3]} {X} {Y} {str(lr)} {str(sg)} {epochs} {int(pca)} {seed} ')
                f.write('%.4lf %.4lf\n'%(somError, top_error))
        # Get the winning neurons for each data point
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
        # print(alldata)
        alldata.to_csv(f"./{profile}/neurons_{nameSOM}.csv", index=False)
    #%%
    if paramsInterest is not None:
        for paramInterest in paramsInterest:
            if paramInterest in paramsname:
                metrics = ["avg", "min", "max"]
                for metric in metrics:
                    paramvalue = np.zeros((X,Y))
                    paramvalue[:,:] = None
                    for i in range(X):
                        for j in range(Y):
                            indexes = alldata[(alldata['Neuron_X']==i) & (alldata['Neuron_Y']==j)].index.tolist()
                            if indexes != []:
                                if metric == "avg":
                                    paramvalue[i,j] = alldata.loc[indexes][paramInterest].mean()
                                elif metric == "min":
                                    paramvalue[i,j] = alldata.loc[indexes][paramInterest].min()
                                else: #metric == "max"
                                    paramvalue[i,j] = alldata.loc[indexes][paramInterest].max()
                    #normSSE = (sseavg-sseavg[sseavg!=-1].min())/(sseavg.max()-sseavg[sseavg!=-1].min())
                    # logSSEavg = np.log(sseavg)
                    # %%
                    # Visualice the SSEmap
                    f = plt.figure(figsize=(10 ,10))
                    ax = f.add_subplot(111)
                    if top == "hex":
                        xx, yy = som.get_euclidean_coordinates()
                        ax.set_aspect('equal')
                        # iteratively add hexagons
                        plt.plot(0, 0)
                        plt.plot(X-1, (Y-1)*np.sqrt(3)/2)
                        if log:
                            logparam = np.log10(paramvalue[np.isnan(paramvalue) != True])
                            minval = logparam.min()
                            deltaval = logparam.max()-minval
                        else:
                            minval = paramvalue[np.isnan(paramvalue) != True].min()
                            deltaval = paramvalue[np.isnan(paramvalue) != True].max()-minval
                        for i in range(X):
                            for j in range(Y):
                                if not np.isnan(paramvalue[i, j]):
                                    color = cm.Blues((np.log10(paramvalue[i, j])-minval)/deltaval)
                                else:
                                    color = "k"
                                wy = yy[(i, j)] * np.sqrt(3) / 2
                                hex = RegularPolygon((xx[(i, j)], wy), 
                                                    numVertices=6, 
                                                    radius=1 / np.sqrt(3),
                                                    facecolor=color)
                                ax.add_patch(hex)
                        # plt.ylim(-1,Y)
                        # plt.xlim(-1,X)
                        xrange = np.arange(X)
                        yrange = np.arange(Y)
                        plt.xticks(xrange-.5, xrange)
                        plt.yticks(yrange * np.sqrt(3) / 2, yrange)

                        divider = make_axes_locatable(plt.gca())
                        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
                        cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, 
                                                    orientation='vertical', alpha=.4)
                        cb1.ax.get_yaxis().labelpad = 100
                        plt.gcf().add_axes(ax_cb)
                    else:#top == "rec"
                        plt.pcolor(np.zeros((X,Y)).T)
                        if log:
                            plt.pcolor(np.log10(paramvalue).T, cmap=cm.Blues)  # Plot the distance_map
                        else:
                            plt.pcolor(paramvalue.T, cmap=cm.Blues)  # Plot the distance_map
                        plt.colorbar()
                        if paramInterest == "SSE":
                            bestperseed = pd.read_csv(f"./Datasets/bestperseed_{profile}__AGA.csv")
                            normalbestperseed = pd.DataFrame()
                            colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
                                'magenta', 'lime', 'teal', 'indigo', 'salmon', 'darkgreen', 'skyblue', 'gold', 'orchid',
                                'sienna', 'lightpink', 'darkgray', 'darkcyan', 'navajowhite', 'rosybrown', 'palevioletred',
                                'mediumseagreen', 'mediumslateblue', 'cadetblue', 'darkgoldenrod', 'thistle', 'dodgerblue', 'khaki',
                                'yellow', 'turquoise']
                            for i in range(7):
                                column = paramsname[i+1]
                                normalbestperseed[column] = (bestperseed[column] - dataunique[column].min()) / (dataunique[column].max() - dataunique[column].min())
                            best_array = normalbestperseed.values
                            for i in range(len(best_array)):
                                wx, wy = som.winner(best_array[i])
                                plt.plot(wx+np.random.random(), wy+np.random.random(), ".", markerfacecolor=colors[i%len(colors)],
                                markeredgecolor=colors[i%len(colors)], 
                                markersize=4, 
                                markeredgewidth=0)
                    plt.suptitle(f"{paramInterest}_{metric}_{nameSOM}")
                    plt.savefig(f"./{profile}/{nameSOM}.png",dpi=169,bbox_inches='tight')
            else:
                print(f"{paramInterest} is no a valid parameter. The valid parameters are the next ones: {paramsname}")
# %%
if __name__ == "__main__":
    #%% Prepare works
                    
    profiles = [
        "1_10_12_6_12",
        "2_10_6_12_6",
        "3_15_12_6_12",
        "4_15_6_12_6"
    ]
    
    SOMs = [
        ["rec", 50, 0.3, 2.9, 10000, 0, 10],
        ["rec", 50, 0.1, 4.0, 150000, 0, 10],
        ["rec", 50, 0.1, 3.6, 150000, 0, 10],
        ["rec", 50, 0.1, 3.9, 150000, 0, 10]
        ]
    #paramsname = ["R_core", "t_Ain", "t_B", "t_Aout", "sAin", "sigmaR", "-log(bg)", "SSE"]
    paramsInterest = ["SSE"]
    use_mp = True 
    log = True
    n_cores = 6
    # %% set works
    works = {}
    w = 0
    for profile in profiles:
        for SOMparameters in SOMs:
            works[w] = {"profile":profile,
                        "SOMparameters":SOMparameters}
            w+=1
    print("Total Works:", w)
    firstwork = 0
    ws = range(firstwork,w)#[0,1,2,3,4,5]
    if use_mp:
        pool = mp.Pool(n_cores)
        partial_work = partial(use_SOM,
                            works = works,
                            paramsInterest = paramsInterest,
                            log = log
                            )
        pool.map(partial_work,[k for k in ws])
        pool.close()
        pool.join
    else: 
        for k in ws:
            use_SOM(k = k ,works = works, paramsInterest = paramsInterest, log = log)