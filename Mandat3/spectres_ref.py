import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

folder = r'C:\Users\jonat\Documents\Polymtl\Session7\PHS3910_e3\Mandat3\spectres_ref'


fig, axs = plt.subplots(8)

for i,filename in enumerate(os.listdir(folder)):
    file = f'{folder}\{filename}'
    df = pd.read_csv(file, header=None).to_numpy().T
    data = np.array([[float(i) for i in str(point).split('\t')] for point in df[0]])
    axs[i].plot(data[:,0], data[:,1])
plt.show()