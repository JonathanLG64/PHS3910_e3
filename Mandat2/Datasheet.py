from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== Find element closest to given target using binary search =====================

def findClosest(arr, target):
    # Returns element closest to target in arr[]
    
    n = len(arr)
    # Corner cases
    if (target <= arr[0]):
        return arr[0]
    if (target >= arr[n - 1]):
        return arr[n - 1]

    # Doing binary search
    i = 0; j = n; mid = 0
    while (i < j):
        mid = (i + j) // 2
 
        if (arr[mid] == target):
            return arr[mid]
 
        # If target is less than array
        # element, then search in left
        if (target < arr[mid]) :
 
            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1]):
                return getClosest(arr[mid - 1], arr[mid], target)
 
            # Repeat for left half
            j = mid
         
        # If target is greater than mid
        else :
            if (mid < n - 1 and target < arr[mid + 1]):
                return getClosest(arr[mid], arr[mid + 1], target)
                 
            # update i
            i = mid + 1
         
    # Only single element left after search
    return arr[mid]

def getClosest(val1, val2, target):
    # Find closest value to target between two values by taking the difference between the target and both values. 
    # It assumes that val2 is greater than val1 and target lies between these two.
 
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1


# ========================== Graphique résolution et contraste ==========================

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
    contrast = np.max(y) - base
    # considère que le contraste est nulle si une mesure invalide est faite
    if np.isnan(contrast):
        contrast = 0
    return [contrast, resolution]

def normalize(sig):
    return sig / np.linalg.norm(sig)

def maxrow(arr):
    indices = np.argmax(arr, axis = 0)
    return np.array(arr[indices[0]])

#shape of r array: (npoints, 3, 14, 1000), shape of s array: (npoints, 3, 1000)
def cont_res_plotter(xvals, s, r, xlabel = 'frequence de coupure [Hz]'):
    x = np.linspace(0,r.shape[2]*2e-2, 14)
    # converti toutes les données en contrastes et résoluitions
    cr =  np.array([[cont_res(x, maxrow([[np.max(np.correlate(normalize(so), normalize(ref), mode='same')) for ref in mes] for mes in pos])) for so in s[n]] for n , pos in enumerate(r)])
    # converti les m en mm pour la résolution
    cr[:,:,1] = cr[:,:,1]*1e3
    # calcule la moyenne et l'écart-type
    cr_m, cr_std = np.mean(cr, axis=1), np.std(cr, axis=1)

    plt.plot(xvals, cr_m[:,0], '-o')
    plt.fill_between(xvals, cr_m[:,0]-cr_std[:,0], cr_m[:,0]+cr_std[:,0], alpha=0.5, color='lightblue')
    plt.xlabel(xlabel)
    plt.ylabel('contraste')
    plt.show()

    plt.plot(xvals, cr_m[:,1], '-o')
    plt.fill_between(xvals, cr_m[:,1]-cr_std[:,1], cr_m[:,1]+cr_std[:,1], alpha=0.5, color='lightblue')
    plt.xlabel(xlabel)
    plt.ylabel('résolution [mm]')
    plt.show()

# ========================== Caractérisation nombre de bits ==========================

# Fonction qui converti les chiffres d'une liste en n bits
def compress(arr, n):
    # Inputs : arr = numpy array, n = nombre de bits
    # Output : numpy array avec chiffres encodés sur n bits

    #numbers = np.linspace(0, (2**n-1), 2**n)
    numbers = np.linspace(min(np.amin(arr)), max(np.amax(arr)), 2**n)
    
    new_array = pd.DataFrame().reindex_like(arr)
    
    # Itterate over all columns of array
    for column in arr:
        values = arr[column]
        
        # Itterate over all values of column and find closest value in x array
        for i in range(len(values)):
            new_array[column][i] = findClosest(numbers, values[i])
            
    return new_array.add_suffix(f"_{n}bits")


# ========================== Caractérisation contenu fréquentil ==========================

def frequency_content(arr, fc):
    pass

if __name__ == '__main__':
    ref = np.reshape(pd.read_csv('table_tests.csv').to_numpy().T, (1,3,14,1000))
    notes =pd.read_csv('table_tests.csv')
    source = np.reshape(notes[['4_1', '4_2', '4_3']].to_numpy().T, (1,3,1000))
    ref = np.reshape(pd.read_csv('old_table_references.csv').to_numpy().T, (3,14,1000))
    notes =pd.read_csv('old_test_table.csv')
    source = np.reshape(notes['g3_3'].to_numpy().T, (1000))
    
    xvals = [1]
    cont_res_plotter(xvals, source, ref, xlabel = 'frequence de coupure [Hz]')
