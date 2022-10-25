import numpy as np
from scipy import signal, io
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter
import seaborn as sns
import pandas as pd

def cont_res(x, y):
    dx = abs(x[0] - x[1])
    peak, _ = signal.find_peaks(y, distance = 1e10)
    width = signal.peak_widths(y, peak, rel_height=0.5)
    resolution = width[0][0]*dx
    contrast = np.max(y) - cont[1][0]
    return [contrast, resolution]

carre = io.loadmat(r'DemiCercle\correl_demi_cercle.mat')
position = io.loadmat(r'Croco\position.mat')
x_plot = io.loadmat(r'Croco\x_plot.mat')

xSource = x_plot['x_plot'][0]
correl = carre['correl']
pos = position['position']


z =  np.array([cont_res(xSource,y) for y in correl])

cont = z[:, 0]
res = z[:, 1]
x = pos[:, 0]
y = pos[:, 1]

X, Y = np.meshgrid(y, x)
Z = np.reshape(cont, (21, 21))

plt.imshow(Z, extent=[min(x), max(x), min(y), max(y)])
plt.show()