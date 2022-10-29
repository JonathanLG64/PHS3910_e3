import numpy as np
from scipy import signal, io
import matplotlib.pyplot as plt

# Calcule le contraste et la résolution d'un signal entré
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
    return [contrast, resolution]

carre = io.loadmat(r'DemiCercle\correl_demi_cercle.mat')
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
# affichage des données dans des heatmaps

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