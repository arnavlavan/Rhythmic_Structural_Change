import pickle
import numpy
import csv
import os

rhythmSC = []

for file in os.listdir():
    if file.endswith(".pkl"):
        templist = []
        templist.append(file[:4])
        songdata = pickle.load(open(file, "rb"))
        templist.append(numpy.mean(songdata['rp_sc_avg']))
        rhythmSC.append(templist)


print(len(rhythmSC))
