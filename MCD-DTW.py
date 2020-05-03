import numpy as np
import librosa
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
import os
from fastdtw import fastdtw




def readmgc(filename):
    # all parameters can adjust by yourself :)
    sr, x = wavfile.read(filename)
    assert sr == 22050
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 256  
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length 
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    print("mgc of {} is ok!".format(filename))
    return mgc



# define your location of your own test data !
natural_folder = 'GT/'
synth_folder = 'baseline/' 
# you need to make sure all waveform files in these above folders have the same file name !


_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
s = 0.0
 
framesTot = 0


files= os.listdir(natural_folder)
for wavID in files:
	print("Processing -----------{}".format(wavID))
	
	filename1 = natural_folder + wavID
	mgc1 = readmgc(filename1)
	filename2 = synth_folder + wavID
	mgc2 = readmgc(filename2)
 
 
	x = mgc1
	y = mgc2


	distance, path = fastdtw(x, y, dist=euclidean)
 
	distance/= (len(x) + len(y))
	pathx = list(map(lambda l: l[0], path))
	pathy = list(map(lambda l: l[1], path))
	x, y = x[pathx], y[pathy]

	frames = x.shape[0]
	framesTot  += frames

	z = x - y
	s += np.sqrt((z * z).sum(-1)).sum()



MCD_value = _logdb_const * float(s) / float(framesTot)

print("MCD = : {:f}".format(MCD_value))



 

