import json
import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from get_intervals import get_pitch_range

intervals = json.load(open("intervals.json"))
plt.figure(figsize=(12, 10))
for prefix, res in intervals.items():
    p1, p2 = get_pitch_range(prefix)
    prefix = prefix[0:-1]
    x, sr = librosa.load(f"{prefix}.aiff", sr=44100)
    

    win = 2048
    pwr = np.cumsum(x**2)
    pwr = pwr[win:] - pwr[0:-win]
    pwr /= np.max(pwr)
    plt.clf()
    plt.plot(pwr)
    for i in res:
        plt.axvline(i, c='k', linestyle='--')
    plt.savefig(f"{prefix}.svg")

    print(prefix, flush=True)
    assert(p2-p1+1 == len(res)-1)
    p = p1
    for i1, i2 in zip(res[0:-1], res[1:]):
        xi = np.array(32767*x[i1:i2], dtype=np.int16)
        fileout = f"{prefix}_p{p}.wav"
        wavfile.write(fileout, sr, xi)
        p += 1
