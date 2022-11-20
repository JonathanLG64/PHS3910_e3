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




#def comb_sinc(x,lbd):   #renvoie le comb*sinc correspondant à notre modèle
#    Lambda = 1e-3/(600)
#
#    u=Lambda/(f2*lbd)
#    v=Lambda*2*np.deg2rad(8.616)/(lbd)
#    w=x[0]
#    b=np.abs(w[0]-w[1])   #step (taille 1 pixel ?)
#    c=1/(Lambda/(f2*lbd))     #step entre les pics du comb
#    d=int(np.round(c/b))      #nombre de pixel entre les pics de dirac
#    DiracComb = np.zeros_like(x)
#    for i in DiracComb:
#        i[::d]= np.sinc(u*w[::d] - v)      #array avec valeur des sinc à la position des pics du comb
#    plt.imshow(DiracComb)
#    plt.show()
#    return DiracComb

def rect(x):
    return np.where(np.abs(x)<=0.5,1,0)

@njit
def comb(x, a):
    step = int(np.round(a / pixel_size))
    diracComb = np.zeros_like(x)
    for diracrow in diracComb:
        diracrow[::step] = 1
    return diracComb

def U2(lbd, f1, f2, a):
    t1 = rect(X*f1/(a*f2))*rect(Y*f1/(b*f2))
    t2 = comb(X,lbd*f2/Lambda)*np.sinc(Lambda*X/(lbd*f2) - Lambda*beta/(2*np.pi))
    return convolve(t1, t2, 'same')

def res(x):
    # Trouve le plus grand pic dans le signal
    try:
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
        intensity = U2(lbd*1e-9, f1, f2, a)
        intensity /= np.max(intensity)
        rgb = wavelength_to_rgb(lbd)
        # multiplies intensities by the rgb values of the wavelength
        rgbdata = U2_to_rgb(intensity, rgb)
        if i == 0:
            combined = rgbdata
        else:
            combined += rgbdata
    return combined
    
if __name__ == '__main__':
    # paramètres constants
    beta = np.radians(8.616) # angle de Blaze
    b = 0.02
    Lambda = (1e-3/(600)) # pas du réseau
    pixel_size = 3.45e-6
    camera_size = (1440, 1080) # 1480 x 1080

    # paramètres à faire varier
    f1 = 50e-3# focale de la 1ere lentille
    f2 = 10e-3#np.array([20, 25, 30, 40, 50])*1e-3# focale de la 2e lentille
    a = 1e-4#np.linspace(0.5e-3, 5e-3, 100)# taille de l'ouverture

    # Définition du domaine spatial
    x = np.linspace(-camera_size[0]*pixel_size/2,
                    camera_size[0]*pixel_size/2,
                    camera_size[0])
    y = np.linspace(-camera_size[1]*pixel_size/2,
                    camera_size[1]*pixel_size/2,
                    camera_size[1])
    X, Y = np.meshgrid(x, y)

    spectrum = get_spectrum(np.linspace(380, 750, 40), f1, f2, a)
    plt.imshow(spectrum)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.show()

    #analyse_f2(550, f1, f2, a)

