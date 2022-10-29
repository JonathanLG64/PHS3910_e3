# Vous devez installer la librairie sounddevice
# PIP :  pip install sounddevice
# Anaconda : conda install -c conda-forge python-sounddevice

import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import wave, pyaudio
import threading
import functools
import pandas as pd
from ast import literal_eval

#from sklearn.model_selection import train_test_split
#from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
#from keras.models import Sequential
#from keras.optimizers import SGD, Adam
#from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from keras.utils import to_categorical
#from sklearn.utils import shuffle

# sélectionne le bon microphone
def setup_mic():
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

def compress(arr):
    return ((2**8 - 1)*(arr - np.min(arr))/np.ptp(arr)).astype(np.uint8)

# memorise une nouvelle commande
def new_command(file = 'memorisedPoints.csv', fs = 44.1e3, seconds=2, chunk_time = 50e-3):
    com = str(input('Nom de la commande associée à la position: '))
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels= 1).T[0]
    sd.wait() 
    chunk = int(chunk_time * fs)
    peak = np.argmax(recording)
    recording = recording[peak:peak+chunk].astype(np.float16)
    try:
        df = pd.read_csv(file)
        df[com] = recording
    except:
        df = pd.DataFrame({com: recording})
    df.to_csv(file, mode = 'w', header = com, index=False)
    t = np.linspace(0, chunk_time, chunk)
    plt.plot(t, recording)
    plt.show()

# définit un décorateur permettant de faire une opération sur un autre thread
def threaded(func):
    """Decorator to automatically launch a function in a thread"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # replaces original function...
        # ...and launches the original in a thread
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

# Joue la note associée au string
@threaded
def play_note(com, p):
    filename=f'Wav-Notes\{com}.wav'
    wf = wave.open(filename)
    chunk = 1024
    stream = p.open(format =
                    p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

def normalize(sig):
    return sig / np.linalg.norm(sig)

# trouve la touche appuyée (prochaine étape: Neural Net)
def get_command(sig, file = 'memorisedPoints.csv'):
    df = pd.read_csv(file)
    corr = [np.max(np.correlate(normalize(sig), normalize(df[col]), mode='same')) for col in df]
    imax = np.argmax(corr)
    prob = corr[imax]
    command = df.keys()[imax]
    print((command, prob))
    if prob < 0.8:
        return 'none'
    return command

# démarre le piano
def run_piano(fs=44.1e3, seconds = 0.1, chunk_time = 50e-3):
    N = int(seconds * fs)
    stream = sd.InputStream(samplerate=fs, channels=1, blocksize=N)
    stream.start()
    audio_streamer = pyaudio.PyAudio()
    chunk = int(chunk_time * fs)
    while True:
        reading = stream.read(frames=N)[0].T[0] #lecture en temps réel
        peak = np.argmax(reading)
        s = reading[peak:peak+chunk].astype(np.float16)
        command = get_command(s)
        if command == 'stop':
            stream.close()
            break
        elif command == 'none':
            pass
        else:
            play_note(command, audio_streamer)

if __name__ == "__main__":
    setup_mic()
    run_piano()
    #new_command()