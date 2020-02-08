import librosa
from scipy.io import wavfile
import pysptk
import numpy as np

 

baseline_file = 'LJ050-0139_gen_baseline.wav'
proposed_file = 'LJ050-0139_gen_GL1.wav'
GT_file = 'LJ050-0139.wav'

 
f1 = open("proposed/139.mgc", "wb")
sr, x = wavfile.read(proposed_file)
print("sr:",str(sr))
#exit()

assert sr == 22050
x = x.astype(np.float64)
#print(x.shape)



frame_length = 1024
hop_length = 256  # 80

# Note that almost all of pysptk functions assume input array is C-contiguous and np.float64 element type
frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T

# Windowing
frames *= pysptk.blackman(frame_length)

assert frames.shape[1] == frame_length 
#print('frames:', frames)




# Order of mel-cepstrum
order = 39
alpha = 0.41
stage = 5
gamma = -1.0 / stage

mgc = pysptk.mgcep(frames, order, alpha, gamma)
print("mgc size::", mgc.size,"    ", mgc.shape)     # mgc size:: 21346   (821, 26) 

mgc = mgc.reshape(-1,1)
print("ccc...", mgc.shape)

mgc = mgc.astype(np.float64)
print("vvvv.....:", mgc.shape)

print("mgc::",mgc)

#mgc = mgc * 0.00000001

print("mgc2::",mgc)

f1.write(mgc)
 
f1.close()
 


print('-----OK!!!------')







