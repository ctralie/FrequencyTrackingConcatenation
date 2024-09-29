import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob
import os
import json

def get_pitch_range(filename):
    c2h = {"C":0, "D":2, "E":4, "F":5, "G":7, "A":9, "B":11}
    c2h.update({f"{key}b":(value-1)%12 for key, value in c2h.items()})
    
    rg = filename.split(".")[-2]
    matches = []
    first = ""
    for key in sorted(c2h.keys()):
        if rg[0:len(key)] == key:
            matches.append(key)
    first = matches[-1]
    octave1 = int(rg[len(first)])
    
    rg = rg[len(first)+1:]
    matches = []
    for key in sorted(c2h.keys()):
        if rg[0:len(key)] == key:
            matches.append(key)
    second = matches[-1]
    octave2 = int(rg[len(second)])
    
    p1 = c2h[first]  + 12*(octave1-4) - 9
    p2 = c2h[second] + 12*(octave2-4) - 9
    return p1, p2

if __name__ == '__main__':
    plt.figure(figsize=(12, 10))
    eps = 1e-2
    intervals = {}
    for dir in glob.glob("*"):
        if not os.path.isdir(dir):
            continue
        for filename in glob.glob(f"{dir}/*"):
            try:
                print(filename)
                x, sr = librosa.load(filename, sr=44100)
                win = 2048
                pwr = np.cumsum(x**2)
                pwr = pwr[win:] - pwr[0:-win]
                pwr /= np.max(pwr)


                zeros = np.array(pwr < eps, dtype=float)
                
                left  = np.where(zeros[1:] - zeros[0:-1] == 1)[0] # Going into silence
                right = np.where(zeros[0:-1] - zeros[1:] == 1)[0] # Coming out of silence
                left = np.concatenate(([0], left))
                right = np.concatenate((right, [pwr.size]))


                # Find the largest intervals based on how many notes there should be
                idx = np.argsort([i1-i2 for i1, i2 in zip(left, right)])
                p1, p2 = get_pitch_range(filename)
                plt.clf()
                plt.plot(pwr)
                idx = idx[0:p2-p1+2]
                res = [int(0.5*(left[i]+right[i])) for i in idx]
                res = list(sorted(res))
                for i in res:
                    plt.axvline(i, c='k', linestyle='--')
                prefix = filename[0:-4]
                intervals[prefix] = res
                plt.savefig(f"{prefix}.svg")
            except:
                print("Exception on", filename)
        json.dump(intervals, open("intervals.json", "w"))