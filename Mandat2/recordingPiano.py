# Vous devez installer la librairie sounddevice
# PIP :  pip install sounddevice
# Anaconda : conda install -c conda-forge python-sounddevice

import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

seconds = 5 
fs = 44100      # Sampling rate    

default = True #Si cette option est utilisée, le micro/speaker par défaut est utilisé
devices = sd.query_devices()

if not default:
    InputStr = "Choisir le # correspondant au micro parmis la liste: \n"
    OutputStr = "Choisir le # correspondant au speaker parmis la liste: \n"
    for i in range(len(devices)):
        if devices[i]['max_input_channels']:
            InputStr += ('%d : %s \n' % (i, ''.join(devices[i]['name'])))
        if devices[i]['max_output_channels']:
            OutputStr += ('%d : %s \n' % (i, ''.join(devices[i]['name'])))
    DeviceIn = input(InputStr)
    DeviceOut = input(OutputStr)

    sd.default.device = [int(DeviceIn), int(DeviceOut)]

print("Recording with : {} \n".format(devices[sd.default.device[0]]['name']))
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels= 1)
sd.wait()

t = np.arange(0,5,1/44100)

plt.plot(t, myrecording)
plt.xlabel('Temps [s]')
plt.ylabel('Amplitude')
plt.show()