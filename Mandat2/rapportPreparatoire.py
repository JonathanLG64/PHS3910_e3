import numpy as np
from scipy import signal, io
import matplotlib.pyplot as plt

def cont_res(x, y):
    dx = abs(x[0] - x[1])
    peak, _ = signal.find_peaks(y, distance = 1e10)
    width = signal.peak_widths(y, peak, rel_height=0.5)
    resolution = width[0][0]*dx
    contrast = np.max(y) - width[1][0]
    return [contrast, resolution]

carre = io.loadmat(r'Carree\carre.mat')
position = io.loadmat(r'Carree\position.mat')
x_plot = io.loadmat(r'Carree\x_plot.mat')

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
Z = np.reshape(cont, (21, 21)).T

plt.imshow(Z, extent=[min(x), max(x), min(y), max(y)])
plt.xlabel('x - position [mm]')
plt.ylabel('y - position [mm]')
plt.colorbar()
plt.show()