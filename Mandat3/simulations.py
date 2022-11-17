import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve, unit_impulse, find_peaks, peak_widths
from numba import njit

def wavelength_to_rgb(wavelength, gamma=0.8):
    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))

def rect(x):
    return np.where(abs(x)<=0.5,1,0)

#TODO: issue where the first step is outside of the camera
def comb(x, a):
    arr = np.zeros_like(x)
    arr[x.shape[0]//2, x.shape[1]//2] = 1

    return arr

def U2(x, y, lbd):
    t1 = rect(x*f1/(a*f2))*rect(y*f1/(b*f2))
    t4 = comb(x, (Lambda/ (f2*lbd))**-1)*np.sinc(Lambda*x / (lbd*f2) - Lambda*beta / (2*np.pi))
    return convolve(t1, t4, mode='same')

def res(x):
    # Trouve le plus grand pic dans le signal
    try:
        plt.plot(x)
        plt.show()
        peak, _ = find_peaks(x, height=0.5)
        # mesure la largeur à mi-hauteur pour calculer la résolution
        width = peak_widths(x, peak, rel_height=0.8)[0][0]
        resolution = width*pixel_size
    except:
        return 0
    return resolution

def res_avg(meshdata):
    avgres = np.mean([res(row) for row in meshdata if res(row) !=0])
    return avgres

def plot_spectrum(X, Y, wavelengths):
    combined = None
    for i, lbd in enumerate(wavelengths):
        intensity = U2(X,Y, lbd*1e-9)
        rgb = np.array(wavelength_to_rgb(lbd))
        # multiplies intensities by the rgb values of the wavelength
        rgbdata = np.array([[(val*rgb).astype(np.uint8) for val in row ] for row in intensity])
        if i == 0:
            combined = rgbdata
        else:
            combined += rgbdata

    plt.imshow(combined)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    wavelengths = [450]# longueurs d'ondes

    # paramètres à définir, d'autres paramètres peuvent intervenir
    f1 = 50e-3# focale de la 1ere lentille
    f2 = 20e-3#np.array([20, 25, 30, 40, 50])*1e-3# focale de la 2e lentille
    a = 2e-4#np.linspace(0.5e-3, 5e-3, 100)# taille de l'ouverture
    beta = np.radians(8.616) # angle de Blaze
    b = 0.02
    Lambda = (1e-3/(600)) # pas du réseau
    pixel_size = 3.45e-6
    camera_size = (1440, 1080) 

    x = np.linspace(-camera_size[0]*pixel_size/2,
                    camera_size[0]*pixel_size/2,
                    camera_size[0])

    y = np.linspace(-camera_size[1]*pixel_size/2,
                    camera_size[1]*pixel_size/2,
                    camera_size[1])
    X, Y = np.meshgrid(x, y)


    plot_spectrum(X, Y, wavelengths)