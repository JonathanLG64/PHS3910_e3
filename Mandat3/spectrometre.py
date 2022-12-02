import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt, peak_widths

def pos_1(row):
    peaks, _ = find_peaks(row, distance=1e4)
    if list(peaks):
        return peaks[0]
    return -1
    
def avg_position1(img):
    return np.mean([pos_1(row) for row in img if pos_1(row) != -1])

def res_pos(row):
    filtrow = medfilt(row, 31)
    peaks, _ = find_peaks(filtrow, prominence=10, distance = 50)
    widths = peak_widths(row, peaks, rel_height=0.5)[0]
    return (peaks, widths)


def avg_pos_res(gray):
    x = []
    r = []
    n = 0
    for row in gray:
        pos, res = res_pos(row)
        if len(pos) > n:
            x = [pos]
            r = [res]
            n = len(pos)
        elif len(pos) == n:
            x.append(pos)
            r.append(res)
    x = np.array(x)
    r = np.array(r)
    r[r<30] = np.nan
    px = 5.2e-6
    pos = np.mean(x, axis=0)
    dpos = np.std(x, axis=0) / pos
    res = np.nanmean(r, axis=0)*px
    dres = np.nanstd(r, axis=0)*px / res
    return (pos, dpos, res, dres)

# interpolation linéaire pour la table optique
def pos_to_lbd_table(x):
    x1 = 123.59701493
    x2 = 1184.526717557252
    y1 = 405
    y2 = 650
    return y1 + (x - x1)*(y2-y1) / (x2-x1)

img = cv2.imread(r'C:\Users\jonat\Documents\Polymtl\Session7\PHS3910_e3\Mandat3\helium.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, dx, r, dr = avg_pos_res(gray) # position et resolution avec erreurs relatives
lbd = pos_to_lbd_table(x)

print(f"longueur d'onde: {lbd}")
print(f"erreur relative longueur d'onde: {dx}")
print(f"résolution (mm): {r*1e3}")
print(f"erreur relative résolution: {dr}")

#lbd =np.array([650, 405])
#dx = np.array([0.01519033, 0.2960821])
#r =np.array([0.00018492, 0.00021245])
#dr =np.array([0.09806197, 0.16599191])

plt.scatter(lbd, r*1e3)
plt.errorbar(lbd, r*1e3, yerr=dr*r*1e3, xerr = lbd*dx, capsize=3, ls='none')

plt.xlabel('ʎ (nm)', fontsize = 15)
plt.ylabel('résolution (mm)', fontsize=15)
plt.show()

plt.imshow(gray)
plt.show()