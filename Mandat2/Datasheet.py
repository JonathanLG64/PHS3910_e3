from scipy import signal
import numpy as np
import pandas as pd

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
    contrast = 1 / base
    print(max(y))
    # considère que le contraste est nulle si une mesure invalide est faite
    if np.isnan(contrast):
        contrast = 0
    return (contrast, resolution)


def plot_cont_res(y1, y2):
    pass

def compress(arr, n):
    pass

def frequency_content(arr, fc):
    pass


if __name__ == '__main__':
    ref = pd.read_csv('table_references.csv')
    notes =pd.read_csv('pianoPoints.csv')
    source = notes[['g3_1', 'g3_2', 'g3_3']]
    print(source)
