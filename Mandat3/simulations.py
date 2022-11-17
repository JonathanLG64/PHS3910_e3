import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve

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

# paramètres à définir, d'autres paramètres peuvent intervenir
f1 = 50e-3# focale de la 1ere lentille
f2 = 20e-3#np.array([20, 25, 30, 40, 50])*1e-3# focale de la 2e lentille
a = 1e-4#np.linspace(0.5e-3, 5e-3, 100)# taille de l'ouverture
Lambda = 1/(600000)# pas du réseau
beta = np.radians(8.616)# angle de Blaze
b = 0.02

pixel_size = 3.45e-6
camera_size = (1440, 1080) 

x = np.linspace(-camera_size[0]*pixel_size/2,
                camera_size[0]*pixel_size/2,
                camera_size[0])

y = np.linspace(-camera_size[1]*pixel_size/2,
                camera_size[1]*pixel_size/2,
                camera_size[1])
X, Y = np.meshgrid(x, y)

def rect(x):
    return np.where(abs(x)<=0.5,1,0)

def comb(x):
    N = x.shape[0]
    arr = np.zeros_like(x)
    arr[N//2, N//2] = 1
    return arr

def U2(x, y, lbd):
    t1 = comb(x*(Lambda/ (f2*lbd)))*np.sinc(Lambda*x / (lbd*f2) - Lambda*beta / (2*np.pi))
    t4 = rect(x*f1/(a*f2))*rect(y*f1/(b*f2))
    return convolve(t1, t4, mode = 'same')

def plot_spectrum(X, Y, wavelengths):
    data = sum([[[wavelength_to_rgb(lbd) * col for col in row] for row in U2(X,Y, lbd*1e-9)] for lbd in wavelengths])
    plt.imshow(data)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


wavelengths = np.linspace(400, 700, 2)# longueur d'onde

plot_spectrum(X, Y, wavelengths)