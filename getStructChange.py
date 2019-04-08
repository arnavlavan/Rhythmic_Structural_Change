
import rp_extract
import audiofile_read
import numpy as np
from matplotlib import __version__  as matplotlib__version__
import matplotlib.pyplot as plt
import pickle
import os

#np.seterr(all='raise')
filepath = 'data/mp3/'
filepath_pkl = 'data/pkl/'
SUMMERY_N = np.array([1,2,4,8,16,32])
PLOT = True

def KL(a,b):
    return np.sum(np.multiply(a,np.log(np.divide(a,b)))) ## AMIR: consider adding  "/ len(a)" for normalization

def calcD(vecA,vecB):
    vecA = vecA / len(vecA)
    vecB = vecB / len(vecB)
    vecAvg = (vecA+vecB) / 2
    KLA = KL(vecA,vecAvg)
    KLB = KL(vecB,vecAvg)
    return (KLA + KLB) / 2

def save2data(_str,data2save):
    global allData
    allData[_str] = data2save
    return True

### MAIN

for filename in os.listdir(filepath):
    if not filename.endswith(".mp3"):
        continue
    else:
        allData = {}
        samplerate, samplewidth, wavedata = audiofile_read.audiofile_read(filepath + filename)


        ### extract rp data
        RP_EXT_N = 6 # 6 -- since the rp_extract works on 6 sec and we want calculation every 1 sec
        for i in range(RP_EXT_N):
            featTmp = rp_extract.rp_extract(wavedata[samplerate*i:], samplerate, extract_rp=True,return_segment_features=True,skip_leadin_fadeout=0)
            save2data('rp_'+str(i), featTmp['rp'])  # <--- this is most of the data
            featSum = np.sum(featTmp['rp'],1)
            if i == 0:
                rpSum = np.zeros(len(featSum)*RP_EXT_N)
            rpSum[i:len(featSum)*RP_EXT_N:RP_EXT_N] = featSum

        itemindex = np.where(rpSum != 0.0)
        rpSum = rpSum[itemindex]
        save2data('rp_sum', rpSum)

        ### calculate sc for each N
        rhytem_sc = np.zeros((len(SUMMERY_N),len(rpSum)))
        for j,sumN in enumerate(SUMMERY_N):
            for i in range(len(rpSum)):
                if i < sumN or len(rpSum) - i <= sumN:
                    continue
                rhytem_sc[j][i] = calcD(rpSum[i-sumN:i+1],rpSum[i:i+sumN+1])


        ### calculate average sc
        goodVals = np.sum(rhytem_sc != 0, 0)
        rhytem_sc_avg = np.zeros(rhytem_sc.shape[1])
        goodIdxs = np.where(goodVals != 0)
        rhytem_sc_avg[goodIdxs] = (np.sum(rhytem_sc, 0))[goodIdxs] / goodVals[goodIdxs]

        save2data('rp_sc', rhytem_sc)
        save2data('rp_sc_N', SUMMERY_N)
        save2data('rp_sc_avg', rhytem_sc_avg)


        filename_pkl = filename.split('.mp3')[0]+'.pkl'
        with open(filepath_pkl + filename_pkl, 'wb') as f:
            pickle.dump(allData, f)
            f.close()

        #### test --> plot sc for all sumN and avg
        if PLOT:
            timeVec = list(range(len(rpSum)))
            plt.plot(np.tile(timeVec,(6,1)).T,rhytem_sc.T)
            if matplotlib__version__ < '3': # hold is deprecated in matplotlib 3.0.0, hold supposed to be always on
                plt.hold
            plt.plot(timeVec, rhytem_sc_avg,c='k',linewidth=3.0)
            lgnd = SUMMERY_N.tolist()
            lgnd.append('avg')
            plt.legend(lgnd)
            plt.show()
