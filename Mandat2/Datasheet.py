from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cont_res(x, y):
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
    contrast = np.max(y) - base
    # considère que le contraste est nulle si une mesure invalide est faite
    if np.isnan(contrast):
        contrast = 0
    return [contrast, resolution]

def normalize(sig):
    return sig / np.linalg.norm(sig)

#shape of r array: (npoints, 3, 14, 1000), shape of s array: (npoints, 1000)
def cont_res_plotter(xvals, s, r, xlabel = 'frequence de coupure [Hz]'):
    x = np.linspace(0,r.shape[2]*2e-2, 14)
    cr =  np.array([[cont_res(x,np.array([np.max(np.correlate(normalize(s[n]), normalize(ref), mode='same')) for ref in mes])) for mes in pos] for n , pos in enumerate(r)])
    cr[:,:,1] = cr[:,:,1]*1e3
    cr_m, cr_std = np.mean(cr, axis=1), np.std(cr, axis=1)

    plt.plot(xvals, cr_m[:,0], '-o')
    plt.fill_between(xvals, cr_m[:,0]-cr_std[:,0], cr_m[:,0]+cr_std[:,0], alpha=0.5, color='lightblue')
    plt.xlabel(xlabel)
    plt.ylabel('contraste')
    plt.show()

    plt.plot(xvals, cr_m[:,1], '-o')
    plt.fill_between(xvals, cr_m[:,1]-cr_std[:,1], cr_m[:,1]+cr_std[:,1], alpha=0.5, color='lightblue')
    plt.xlabel(xlabel)
    plt.ylabel('résolution [mm]')
    plt.show()

def compress(arr, n):
    pass

def frequency_content(arr, fc):
    pass

if __name__ == '__main__':
    ref = np.reshape(pd.read_csv('table_references.csv').to_numpy().T, (1,3,14,1000))
    notes =pd.read_csv('test_table.csv')
    source = np.reshape(notes['e3_3'].to_numpy().T, (1,1000))
    xvals = [1, 2]
    cont_res_plotter(xvals, source, ref)
