
print("\nStarting analysis, please wait...")

import librosa
import matplotlib.pyplot as plt
from librosa import display
import time

def write_to_file(second_db,talk_counter):
    counter = 1
    with open('assets/docs/Talktime.csv','w') as f:
        for i in second_db:
            f.writelines(f'\n{counter},{i}')
            counter+=1
        f.writelines(f'\n\nTalktime:,{talk_counter} seconds')

#------------------------------------------------------------------------------

audio = 'assets/audio/audio.wav'
x, sr = librosa.load(audio)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize = (10, 5))
librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
plt.colorbar()
plt.savefig("assets/audiodocs/spectrogram.pdf", format="pdf", bbox_inches="tight")
print("Spectogram exported to PDF...")
#------------------------------------------------------------------------------

import sklearn

spectral_centroids = librosa.feature.spectral_centroid(y = x, sr = sr)[0]

# Computing the time variable for visualization
plt.figure(figsize = (12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis = 0):
  return sklearn.preprocessing.minmax_scale(x, axis = axis)

#Plotting the Spectral Centroid along the waveform
librosa.display.waveshow(x, sr = sr, alpha = 0.4)
plt.plot(t, normalize(spectral_centroids), color = 'b')
plt.savefig("assets/audiodocs/waveform.pdf", format="pdf", bbox_inches="tight")
print("Waveform exported to PDF...")
#-------------------------------------------------------------------------------

# Calculating amplitude for each second of audio file
import numpy as np

file = "assets/audio/audio.wav"
y, sr = librosa.load(file)
second = []
for s in range(0,len(y),sr):
    second.append( np.abs(y[s:s+sr]).mean())

# Calculating amplitude of a single second of silence
file_silence = "assets/audio/silent_1-second.wav"
y, sr = librosa.load(file_silence)
second_silence = []
for s in range(0,len(y),sr):
    second_silence.append( np.abs(y[s:s+sr]).mean())

# Calculating decibel level of each second of audio file relative to silence
import math

amp_ref = second_silence
second_db = list()

for amp in second:
  power_db = 20 * math.log10(amp / amp_ref)
  second_db.append(power_db)

# Caculating total seconds where audio exceeded a fixed threshold decibel value
threshold = 75
talk_counter = 0

for db in second_db:
  if db > threshold:
    talk_counter+=1

write_to_file(second_db,talk_counter)
print("CSV generated. Exiting...")
time.sleep(5)
