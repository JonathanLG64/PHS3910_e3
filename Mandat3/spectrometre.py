import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks, medfilt, peak_widths
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

mpl.rcParams['figure.dpi'] = 150
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def gaussian(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2 / (2*sigma**2))
# applique des régressions gaussiennes sur tous les pics trouvés dans le signal
def gaussfitter(row):
    row = medfilt(row, 31)
    peaks, _ = find_peaks(row, prominence=10, distance = 50)
    width = peak_widths(row, peaks, rel_height=0.5)
    pos = []
    errs = []
    pixels = np.arange(row.size)
    xdata, err, popt = to_lbd(pixels)
    xerr = pixels*(popt[0] + err[0]) + err[1] + popt[1] - xdata

    plt.plot(xdata, row, '-',label ='signal filtré', alpha=0.3)

    for i , peak in enumerate(peaks):
        
        roi = np.logical_and(xdata > to_lbd(width[2][i])[0], xdata < to_lbd(width[3][i])[0])
        popt, pcov = curve_fit(gaussian, 
                               xdata[roi],
                               row[roi], p0=[127, to_lbd(float(peak))[0], 1])
        err = np.sqrt(np.diag(pcov))

        print(popt, err)
        print(f'resolution = {popt[2]/2.355:.3f} +- {err[2]/2.355:.3f} nm')

        plt.plot(xdata[roi], gaussian(xdata[roi], *popt), '--' ,label=f'fit gaussien {i+1}')
        plt.plot(xdata[peak], row[peak],'.')
        pos.append(popt[1])
        errs.append(popt[2])
    
    plt.xlabel('λ (nm)', fontsize = 15)
    plt.ylabel('Intensité', fontsize=15)
    plt.legend()
    plt.show()

    return (np.array(pos), np.array(errs)/np.array(pos))

def linear(x, a, b):
    return a*x + b
#calibration de l'appareil (droite linéaire qui convertit des pixels en nm)
def to_lbd(x):
    #table optique
    #x_ref = np.array([ 214.46598825, 588.77195384, 954.24376857, 1127.0321855 ])
    #impression 3D
    x_ref = np.array([ 363.55249548,  703.86088748,  874.53796053, 1135.02687714])

    y_ref = np.array([389, 444.4, 497.9, 581.5])

    popt, pcov = curve_fit(linear, x_ref, y_ref, p0=[0.2, 337])
    err = np.sqrt(np.diag(pcov))
    #x1 = np.linspace(0, 1280, 1000)
    #plt.scatter(x_ref, y_ref, label='points de référence')
    #plt.plot(x1, linear(x1,*popt), label='droite de calibration')
    #plt.xlabel('x (pixels)', fontsize=15)
    #plt.ylabel('λ (nm)', fontsize=15)
    #plt.legend()
    #plt.fill_between(x1, linear(x1,*popt) - err[0]*x1 
    # -err[1],linear(x1,*popt) + err[0]*x1 +err[1], alpha=0.3, color='blue')
    #plt.show()
    #print(popt)
    #print(err)
    y = popt[0]*x + popt[1]
    return (y, err, popt)
    
#img = cv2.imread(r'C:\Users\jonat\Documents\Polymtl
# \Session7\PHS3910_e3\Mandat3\Spectres_cond_reels\helium.png')
file = r'~/Documents/TrimestreA22/PHS3910/PHS3910_e3/Mandat3/Spectres_cond_reels/helium.png'
img = cv2.imread(file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x, errors = gaussfitter(gray[int(1080*0.5)])