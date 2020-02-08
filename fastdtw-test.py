import numpy as np
import librosa
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


 
framesTot = 0
natural_tag = 'GT'
synth_tag = 'baseline'

indexf = open('uttlist.txt')
indexfile = indexf.readlines()


def readmgc(filename):
    sr, x = wavfile.read(filename)
    assert sr == 22050
    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 256  # 80

    # Note that almost all of pysptk functions assume input array is C-contiguous and np.float64 element type
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T

    # Windowing
    frames *= pysptk.blackman(frame_length)

    assert frames.shape[1] == frame_length 
    #print('frames:', frames)

    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1,order+1)
    return mgc



_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
s = 0.0

for uttID in indexfile:
    print("Processing -----------------",str(uttID) )
    uttID = uttID.strip('\n')
    filename1 = natural_tag + '/' + uttID + '.wav'
    
    filename2 = synth_tag + '/' + uttID + '.wav'
    print("filename1::",filename1)
    print("filename2::",filename2)

    x = readmgc(filename1)
    y = readmgc(filename2)
 

    #x = np.reshape(feature1, (-1, 26))
    #y = np.reshape(feature2, (-1, 26))
    print("x::",x.shape)
    print("y::",y.shape)

    distance, path = fastdtw(x, y, dist=euclidean)
    #dist, path = fastdtw(x, y, radius=self.radius, dist=self.dist)
    distance/= (len(x) + len(y))
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    x, y = x[pathx], y[pathy]
    print("x size:", x.shape)
    print("y size:", y.shape)

    frames = x.shape[0]
    framesTot  += frames

    z = x - y
    s += np.sqrt((z * z).sum(-1)).sum()



MCD_value = _logdb_const * float(s) / float(framesTot  )

print("MCD = : %f" % MCD_value)
 


