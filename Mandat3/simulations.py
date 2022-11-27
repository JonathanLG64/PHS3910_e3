import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
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
    return np.where(np.abs(x)<=0.5,1,0)

def comb(x, a):
    step = int(np.round(a / pixel_size))
    diracComb = np.zeros_like(x)
    diracComb[:, ::step] = 1 
    return diracComb

def U2(lbd, f1, f2, a):
    beta = np.arcsin(lbd/Lambda - np.sin(θ_i))
    kappa = 0#2*np.pi*(np.sin(θ_i) + np.sin(beta)) / lbd
    t1 = rect(X*f1/(a*f2))*rect(Y*f1/(b*f2))
    t2 = comb(X,lbd*f2/Lambda)*np.sinc(Lambda*(X/(lbd*f2) - kappa/(2*np.pi)))
    t3 = convolve(t1, t2, 'same')
    return t3

@njit    
def U2_to_rgb(data, rgb):
    rgbdata = np.zeros((data.shape[0], data.shape[1], 3), dtype = np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(3):
                rgbdata[i,j,k] = data[i,j]*rgb[k]
    return rgbdata

def get_spectrum(wavelengths, f1, f2, a):
    combined = None
    for i, lbd in enumerate(wavelengths):
        intensity = np.abs(U2(lbd*1e-9, f1, f2, a))
        intensity /= np.max(intensity)
        rgb = wavelength_to_rgb(lbd)
        # multiplies intensities by the rgb values of the wavelength
        rgbdata = U2_to_rgb(intensity, rgb)

        if i == 0:
            combined = rgbdata
        else:
            combined += rgbdata
        print(f'Loading... {i*100/wavelengths.size:.2f}% done')
    return combined
    
if __name__ == '__main__':
    θ_b = np.radians(8.616) # angle de Blaze
    θ_i = np.radians(50) # angle d'incidence
    b = 1e-1
    Lambda = (1e-3/(600)) # pas du réseau
    pixel_size = 5.2e-6
    camera_size = (4000, 1080) # 1480 x 1080

    f1 = 50e-3# focale de la 1ere lentille
    f2 = 30e-3#np.array([20, 25, 30, 40, 50])*1e-3# focale de la 2e lentille
    a = 1e-4#np.linspace(0.5e-3, 5e-3, 100)# taille de l'ouverture

    # Définition du domaine spatial
    x = np.linspace(-camera_size[0]*pixel_size/2,
                    camera_size[0]*pixel_size/2,
                    camera_size[0])

    y = np.linspace(-camera_size[1]*pixel_size/2,
                    camera_size[1]*pixel_size/2,
                    camera_size[1])

    X, Y = np.meshgrid(x, y)

    spectrum = get_spectrum(np.linspace(400, 700, 10), f1, f2, a)

    plt.imshow(spectrum, extent=[min(x)*1e3, max(x)*1e3, min(y)*1e3, max(y)*1e3])
    plt.xlabel('x (mm)', fontsize=15)
    plt.ylabel('y (mm)', fontsize=15)
    plt.show()

