# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
algorithms = [
    "AGA",
    # "GHS00",
    # "GHS"
    ]
info = "" #added in the file's name
Iexps = [
    "1_10_12_6_12",
    # "2_10_6_12_6",
    "3_15_12_6_12",
    "4_15_6_12_6"
    ]
minvalu = np.array([50, 30, 30, 30, 0.1, 0.0, 2.5])
maxvalu = np.array([250, 200, 200, 200, 0.45, 0.45, 5.5])
expectv = {"1_10_12_6_12_0.5_3.8" : [100, 120, 60, 120, 0.2, 0.2, 3.8],
           "2_10_6_12_6_0.7_4" : [100, 60, 120, 60, 0.2, 0.2, 4],
           "3_15_12_6_12_0.55_4.2" : [150, 120, 60, 120, 0.2, 0.2, 4.2],
           "4_15_6_12_6_0.8_4" : [150, 60, 120, 60, 0.2, 0.2, 4]}
paramsname = ["seed","R_core", "t_Ain", "t_B", "t_Aout", "sAin", "sigmaR", "-log(bg)", "SSE"]
            

# %% Parameters evolution
def distnorm(Iexp, params):
    d = len(params)
    normval = np.array([30, 30, 30, 30, 0.1, 0.1])
    normparam = (params-expectv[Iexp][:d])/normval[:d]#/(maxvalu[:d]-minvalu[:d]) 
    distNorm = np.linalg.norm(normparam)/np.sqrt(d)
    return distNorm
for Iexp in Iexps:
    for alg in algorithms:
        allparams = np.zeros((1,8))
        bestparams = np.zeros((1,9),dtype=object)
        directorio = f"/Users/Diego Felipe Ramirez/Downloads/Documents/U/TG/code/SOM/Data/{alg}/{Iexp}"#/{n}"
        # Verificar si la ruta es un directorio válido
        if os.path.isdir(directorio):
            carpetas = [nombre for nombre in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, nombre))]
            tempath = f"./Data/{alg}/{Iexp}"#/{n}"
            if alg == "AGA":
                for name in carpetas:
                    tempPath = f"{tempath}/{name}"
                    maxIter = int(np.genfromtxt(f'{tempPath}/current_cicle.txt'))
                    for i in range(maxIter):
                        if os.path.isfile(f"{tempPath}/results_{i}.txt"):
                            tempdb = np.loadtxt(f"{tempPath}/results_{i}.txt",skiprows=1)
                            tempparams = tempdb[:,1:8]
                            tempsse = np.array([tempdb[:,-1]]).T
                            tempparams = np.append(tempparams, tempsse, axis=1)
                            allparams = np.append(allparams, tempparams, axis=0)
                            if i == 99:
                                besti = np.argmin(tempdb[:,-1])
                                bestparam = np.append(int(name.split("_")[1][1:]), tempparams[besti, :])
                                bestparams = np.append(bestparams, [bestparam], axis=0)
            else: #GHS or GHS00
                for name in carpetas:
                    tempPath = f"{tempath}/{name}"
                    if os.path.isfile(f"{tempPath}/all_harmonies.txt"):
                        tempdb = np.loadtxt(f"{tempPath}/all_harmonies.txt",skiprows=1)
                        tempparams = tempdb[:,1:9]
                        allparams = np.append(allparams, tempparams, axis=0)
            allparams = allparams[1:,:]
            bestparams = bestparams[1:,:]
            np.savetxt(f'{Iexp}_{info}_{alg}.csv', allparams, delimiter=',', header=','.join(paramsname[1:]), comments='')
            np.savetxt(f'bestperseed_{Iexp}_{info}_{alg}.csv', bestparams, delimiter=',', header=','.join(paramsname), comments='')
        else:
            print(f'La ruta "{directorio}" no es un directorio válido.')
        