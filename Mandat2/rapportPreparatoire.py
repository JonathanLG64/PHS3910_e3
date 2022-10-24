import numpy as np
from scipy import signal, io
import matplotlib.pyplot as plt

file = 'random_shape_data.mat'
data = io.loadmat(file)

def cont_res(x, y):
    try:
        dx = abs(x[0] - x[1])
        peak, _ = signal.find_peaks(y, distance = 1e10)
        width = signal.peak_widths(y, peak, rel_height=0.5)
        resolution = width[0][0]*dx
        contrast = np.max(y) - width[1][0]
        return (contrast, resolution)
    except:
        return (np.inf, np.inf)

x = data['abs_x'][0]
y = data['correl_y'][0]

cont, res = cont_res(x,y)
print(cont, res)

