import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import wave, pyaudio
import threading
import functools
import pandas as pd

def setup_mic():
    """Selects the microphone to be used for measurements"""
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

def new_command(file = 'pianoPoints.csv', fs = 44.1e3, seconds=2, chunk_time = 50e-3):
    """ Saves a new reference point in the database and plots it.
    input parameters:
        file-> name of file containing the database
        fs-> sampling frequency
        seconds-> amount of time given to take the measurement
        chunk_time-> duration of data saved in the database
    """
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

#https://stackoverflow.com/questions/67071870/python-make-a-function-always-use-a-thread-without-calling-thread-start
def threaded(func):
    """Decorator to automatically launch a function in a thread"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # replaces original function...
        # ...and launches the original in a thread
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

@threaded
def play_note(com, p):
    """Plays the note assiciated to the given string
    input parameters:
        com-> a string containing the name of the note to be played
        p-> audio streamer object
    """
    file = com.split('_')
    filename=f'Wav-Notes\{file[0]}.wav'
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
    """Normalizes the signal"""
    return sig / np.linalg.norm(sig)

def get_command(sig, file = 'pianoPoints.csv'):
    """Finds the piano tile that was pressed by using the maximum of correlation
    Input parameters:
        sig-> signal that was detected
        file-> name of the file containing the database
               of the previous measurements that were done
    Output parameters:
        command-> the command that had the best correlation
    """
    df = pd.read_csv(file)
    corr = np.array([np.max(np.correlate(normalize(sig),
     normalize(df[col]), mode='same')) for col in df])
    imax = np.argmax(corr)
    prob = corr[imax]
    command = df.keys()[imax]
    if prob < 0.83:
        return 'none'
    print((command, prob))
    return command

def run_piano(fs=44.1e3, seconds = 0.1, chunk_time = 50-3, file = 'pianoPoints.csv'):
    """Runs the piano in real time
    Input parameters:
        fs-> fréquence d'échantillonage
        seconds-> minimum time between 
        chunk_time-> time used to compare the signals
        file-> the name of the file containing the database
    """

    N = int(seconds * fs)
    stream = sd.InputStream(samplerate=fs, channels=1, blocksize=N)
    stream.start()
    audio_streamer = pyaudio.PyAudio()
    chunk = int(chunk_time * fs)
    while True:
        reading = stream.read(frames=N)[0].T[0]
        peak = np.argmax(reading)
        s = reading[peak:peak+chunk].astype(np.float16)
        command = get_command(s, file=file)
        if command == 'stop':
            stream.close()
            break
        elif command == 'none':
            pass
        else:
            play_note(command, audio_streamer)

if __name__ == "__main__":
    ct = 50e-3
    fs = 20e3
    f = 'banque_reference_notes.csv'
    setup_mic()
    run_piano(chunk_time=ct, fs=fs, file=f)
    #Used to save new points in the dataset in a speedy way
    #while True:
    #    new_command(chunk_time=ct, fs=fs, file=f)