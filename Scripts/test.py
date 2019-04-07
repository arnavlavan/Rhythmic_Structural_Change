
from rp_extract import rp_extract
from rp_extract import audiofile_read
import numpy as np
import matplotlib.pyplot as plt

### ONE SONG ONLY --> Need to run on all songs
#samplerate, samplewidth, wavedata = audiofile_read.audiofile_read("../data/0074 omer adam - no'etzet mabat.mp3")
samplerate, samplewidth, wavedata = audiofile_read.audiofile_read("C:/Users/OWNER/Dropbox/1phd/database/files/0666 aviv geffen - achshav meunan.mp3")


# 4 samples from left and right ---> should be changed to 1,2,4,8,16,32 and after preform "mean" on all vectors (note for the zeros in the beginnning and the end)
SUMMERY_N = 4


def KL(vec):
    M = len(vec)
    return np.sum(np.multiply(vec,np.log(vec/M)))

def calcD(vecA,vecB):
    KLA = KL(vecA)
    KLB = KL(vecB)
    return (KLA + KLB) / 2


for i in range(6):
    featTmp = rp_extract.rp_extract(wavedata[samplerate*i:], samplerate, extract_rp=True, extract_ssd=True, extract_rh=True,return_segment_features=True,skip_leadin_fadeout=0)
    featSum = np.sum(featTmp['rp'],1)
    if i == 0:
        featAll = np.zeros(len(featSum)*6)
        timeVec = list(range(len(featSum)*6))
    featAll[i:len(featSum)*6:6] = featSum


rhytem_sc = np.zeros(len(featAll))
for i in range(len(featAll)):
    if i < SUMMERY_N or len(featAll) - i < SUMMERY_N:
        continue
    rhytem_sc[i] = calcD(featAll[i-SUMMERY_N:i+1],featAll[i:i+SUMMERY_N+1])

plt.plot(timeVec,rhytem_sc)
plt.show()
