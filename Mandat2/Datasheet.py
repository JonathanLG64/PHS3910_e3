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

def maxrow(arr):
    indices = np.argmax(arr, axis = 0)
    return arr[indices[0]]

#shape of r array: (npoints, 3, 14, 1000), shape of s array: (npoints, 3, 1000)
def cont_res_plotter(xvals, s, r, xlabel = 'frequence de coupure [Hz]'):
    x = np.linspace(0,r.shape[2]*2e-2, 14)
    # converti toutes les données en contrastes et résoluitions
    cr =  np.array([[maxrow([cont_res(x,np.array([np.max(np.correlate(normalize(so), normalize(ref), mode='same')) for ref in mes])) for mes in pos]) for so in s[n]]for n , pos in enumerate(r)])
    print(cr)
    #print(cr.shape)
    cr[:,:,1] = cr[:,:,1]*1e3 # converti les m en mm pour la résolution

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

# Fonction qui converti les chiffres d'une liste en n bits
def compress(arr, n):
    # Inputs : arr = numpy array, n = nombre de bits
    # Output : numpy array avec chiffres encodés sur n bits

    x = np.linspace(0, (2**n-1), 2**n)
    pass

def frequency_content(arr, fc):
    pass

if __name__ == '__main__':
    ref = np.reshape(pd.read_csv('table_references.csv').to_numpy().T, (1,3,14,1000))
    notes =pd.read_csv('table_references.csv')
    source = np.reshape(notes[['5_1', '5_2', '5_3']].to_numpy().T, (1,3,1000))
    ref = np.reshape(pd.read_csv('old_table_references.csv').to_numpy().T, (3,14,1000))
    notes =pd.read_csv('old_test_table.csv')
    source = np.reshape(notes['g3_3'].to_numpy().T, (1000))
    
    #xvals = [1]
    #cont_res_plotter(xvals, source, ref, xlabel = 'frequence de coupure [Hz]')
