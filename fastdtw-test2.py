import numpy as np
import librosa
from scipy.io import wavfile
import pysptk
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw




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
    order = 39
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1,40)
    return mgc



framesTot = 0
natural_tag = 'Odyssey-speech-samples/GT-wavenet/'
synth_tag = 'Odyssey-speech-samples/baseline-wavenet/'

indexf = open('wavlist.txt')
indexfile = indexf.readlines()

num=0
for wavID in indexfile:
    num = num + 1
    print("----",str(num))
    print("Processing -----------------",str(wavID) )
    wavID = wavID.strip('\n')
    filename1 = natural_tag + wavID.replace('.','_gen.')
     
    mgc1 = readmgc(filename1)
    print("mgc1 is ok!")
    filename2 = synth_tag + wavID.replace('.','_gen.')
     
    mgc2 = readmgc(filename2)
    print("mgc2 is ok!")


    #print("filename1::",filename1)
    #print("filename2::",filename2)


    #f1 = open(filename1, "rb")
    #feature1 = np.fromfile(f1, dtype=np.float64)
    #f1.close()

    #f2 = open(filename2, "rb")
    #feature2 = np.fromfile(f2, dtype=np.float64)
    #f2.close()

    #x = np.reshape(feature1, (-1, 40))
    #y = np.reshape(feature2, (-1, 40))
    #print("x::",x.shape)
    #print("y::",y.shape)

    distance, path = fastdtw(mgc1, mgc2, dist=euclidean)
    print("distance:", distance)
    frames = mgc1.shape[0]

    minCostTot += distance
    framesTot  += frames

print("score:",minCostTot / framesTot)
print('overall MCD = %f (%d frames)' % (minCostTot / framesTot, framesTot ))


