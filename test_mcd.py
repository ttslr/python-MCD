import mcd
import struct
import numpy as np
import pickle
 
print("==========111111111111===============")
f = open('GT/139.mgc', "rb")

feature = np.fromfile(f, dtype=np.float64)

f.close()
print("feature :", feature)
print("feature size:", feature.size, "      ", feature.shape)  #  24520        (24520,)

print("==========2222222222===============")
f2 = open('baseline/cmu_us_arctic_slt_a0044.mgc', "rb")

feature2 = np.fromfile(f2, dtype=np.float64)

f2.close()
print("feature2 :", feature2)
print("feature2 size:", feature2.size, "      ", feature2.shape)  #  24520        (24520,)
 





