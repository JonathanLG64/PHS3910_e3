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

# memorise une nouvelle commande
def new_command(fs = 44100, seconds=3, chunk_time = 1000e-3):
    chunk = int(chunk_time * fs)
    com = str(input('Nom de la commande associée à la position: '))
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels= 1)
    sd.wait()
    n = myrecording.size//2
    d = {'command' : com, 'S(t)': [myrecording[n:n+chunk].T.tolist()]}
    df = pd.DataFrame(data=d)
    with open('memorisedPoints.csv', 'a') as f:
        df.to_csv(f, mode='a', index=False, header=f.tell()==0)

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
def get_command(sig):
    df = pd.read_csv('memorisedPoints.csv')
    commands , ms = list(df['command']), list(df['S(t)'])
    correlations = [(com, np.max(np.abs(np.correlate(normalize(sig), 
    normalize(np.array(literal_eval(s))[0]))))) for com, s in zip(commands, ms)]
    top = max(correlations, key=lambda x: x[1])
    print(top)
    return top[0]

# démarre le piano
def run_piano(fs=44100, chunk_time = 1000e-3):
    n = int(chunk_time * fs)
    stream = sd.InputStream(samplerate=fs, channels=1, blocksize=n)
    stream.start()
    audio_streamer = pyaudio.PyAudio()
    while True:
        reading = stream.read(frames=n)[0].T[0] #lecture en temps réel
        command = get_command(reading)
        if command == 'stop':
            stream.close()
            break
        elif command == 'none':
            pass
        else:
            play_note(command, audio_streamer)

if __name__ == "__main__":
    setup_mic()
    #run_piano()
    new_command()
    