import numpy as np
from scipy import signal, io
import matplotlib.pyplot as plt

def cont_res(x, y):
    dx = abs(x[0] - x[1])
    peak, _ = signal.find_peaks(y, distance=10e3)
    width = signal.peak_widths(y, peak, rel_height=0.5)[0][0]
    resolution = width*dx
    contwidth = signal.peak_widths(y, peak, rel_height=0.8)
    left = int(contwidth[2][0])
    right = int(contwidth[3][0])
    base = (np.mean(y[x < x[left]]) + np.mean(y[x > x[right]]))/2
    contrast = np.max(y) / base
    if np.isnan(contrast):
        contrast = 0
    return [contrast, resolution]

carre = io.loadmat(r'Croco\croco_correl.mat')
position = io.loadmat(r'DemiCercle\position.mat')
x_plot = io.loadmat(r'DemiCercle\x_plot.mat')

xSource = x_plot['x_plot'][0]
correl = carre['correl']
pos = position['position']


z =  np.array([cont_res(xSource,y) for y in correl])

cont = z[:, 0]
res = z[:, 1]
x = pos[:, 0]*1e3
y = pos[:, 1]*1e3

x = x -100
y = y - 100
X, Y = np.meshgrid(x, y)
Zres = np.reshape(res, (21, 21)).T*1e3
Zcont = np.reshape(cont, (21, 21)).T

plt.imshow(Zres, extent=[min(x), max(x), min(y), max(y)])
plt.xlabel('x - position [mm]')
plt.ylabel('y - position [mm]')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Résolution [mm]', rotation=90, labelpad=15)
plt.show()

plt.imshow(Zcont, extent=[min(x), max(x), min(y), max(y)])
plt.xlabel('x - position [mm]')
plt.ylabel('y - position [mm]')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Contraste', rotation=90, labelpad=15)
plt.show()