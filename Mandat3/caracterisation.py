import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from numba import njit

# ======== Find element closest to given target using binary search =========

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
    # Find closest value to target between two values 
    # by taking the difference between the target and both values. 
    # It assumes that val2 is greater than val1 and target lies between these two.
 
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1

# ======== Find wavelenght =========

def position_to_lambda(intensity_df):
    '''
    Input:
        intensity_df: matrice de la taille du nombre de pixel de la caméra (1280 x 1080) avec les valeurs d'intensité en fonction de la position (x, y)

    Output: 
        lambda: Longueur d'onde correspondant à la matrice d'intensité


    '''
# Faire dictionnaire avec longueur d'onde (clé) et le gaz (valeurs)
# Function mm_to_lamba qui convertie la positon en mm en une longueur d'onde 
# (faire une calibrationo avec 400 nm et 700nm avec leur position) et pour les autres valeurs entre 400 et 700 on fait une interpoolation linéaire
# Prend en input matrice de la taille camera size avec des valeurs d'intensité 