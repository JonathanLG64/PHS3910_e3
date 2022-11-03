from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cont_res(x, y):
    print(y.shape)
    dx = abs(x[0] - x[1])
    # Trouve le plus grand pic dans le signal
    peak, _ = signal.find_peaks(y, distance=1e3)
    # mesure la largeur à mi-hauteur pour calculer la résolution
    width = signal.peak_widths(y, peak, rel_height=0.5)[0][0]
    resolution = width*dx
    # Fait la moyenne des points qui sont considérés à l'extérieur du pic
    contwidth = signal.peak_widths(y, peak, rel_height=0.8)
    left = int(contwidth[2][0])
    right = int(contwidth[3][0])
    # calcule la moyenne de ces points comme baseline
    base = (np.mean(y[x < x[left]]) + np.mean(y[x > x[right]]))/2
    contrast = 1 / base
    print(max(y))
    # considère que le contraste est nulle si une mesure invalide est faite
    if np.isnan(contrast):
        contrast = 0
    return [contrast, resolution]

def normalize(sig):
    return sig / np.linalg.norm(sig)

#shape of r array: (npoints, 3, 14, 1000), shape of s array: (npoints, 1000)
def cont_res_plotter(x, s, r):
    correlations =  np.array([[[[np.max(np.correlate(normalize(source), normalize(ref), mode='same')) for ref in mes] for mes in pos] for pos in r] for source in s])
    for i in range(3):
        plt.plot(x, correlations[0,0,i,:])
    plt.xlabel('x [m]')
    plt.ylabel('correlation')
    plt.show()
    pass

def compress(arr, n):
    pass

def frequency_content(arr, fc):
    pass

if __name__ == '__main__':
    ref = np.reshape(pd.read_csv('table_references.csv').to_numpy().T, (1,3,14,1000))
    notes =pd.read_csv('table_references.csv')
    source = np.reshape(notes['7_3'].to_numpy().T, (1,1000))
    x = np.linspace(0,ref.shape[2]*2e-2, 14)
    print(ref)
    print(x)
    cont_res_plotter(x, source, ref)
