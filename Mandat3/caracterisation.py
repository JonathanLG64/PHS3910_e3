import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt

def pos_1(row):
    peaks, _ = find_peaks(row, distance=1e4)
    if list(peaks):
        return peaks[0]
    return -1
    
def avg_position1(img):
    return np.mean([pos_1(row) for row in img if pos_1(row) != -1])

def pos_multi(row):
    filtrow = medfilt(row, 31)
    peaks, _ = find_peaks(filtrow, prominence=10, distance = 50)
    return peaks

def avg_position(gray):
    x = []
    n = 0
    for row in gray:
        pos = pos_multi(row)
        if len(pos) > n:
            x = [pos]
            n = len(pos)
        elif len(pos) == n:
            x.append(pos)
    x = np.array(x)
    return np.mean(x, axis=0)

# interpolation linéaire pour la table optique
def pos_to_lbd_table(x):
    x1 = 145.15453194650817
    x2 = 1184.526717557252
    y1 = 405
    y2 = 650
    return y1 + (x + x1)*(y2-y1) / (x2-x1)

img = cv2.imread(r'C:\Users\jonat\Documents\Polymtl\Session7\PHS3910_e3\Mandat3\helium.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x = avg_position(gray)
lbds = pos_to_lbd_table(x)
print(lbds)

plt.imshow(gray)
plt.show()