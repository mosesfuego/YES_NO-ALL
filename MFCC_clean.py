#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:23:33 2020

@author: codeplus
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
import librosa as lb
import pandas as pd
from glob import glob
import os  
from scipy import stats
from scipy import signal


folder = "clean"
# Read C_1.wav file
Fs = 16000
freq_cutoff = 1000
featureNum = 25

wav_paths = glob('{}/**'.format(folder), recursive=True)
file_count = len(wav_paths) #- len(os.listdir(folder)) - 1 + 4
print(file_count)
import os

list = os.listdir(folder) # dir is your directory path
number_files = len(list)
print (number_files)


classes = os.listdir(folder)
print(len(classes))
instLabel = 0
target = np.zeros(file_count, dtype=int).T
targIter = 0
features = np.zeros([file_count, featureNum])


for cls in classes:
    wavs = os.listdir(folder + "/" + cls)
    for wav in wavs:
        file = folder + "/" + cls + "/" + wav
        audio_data, Fs = lb.load(file, sr=Fs)
        y = audio_data
        sr = Fs
        MFCC_sequence = lb.feature.mfcc(audio_data, sr=Fs, n_mfcc=featureNum, dct_type=2)
    
        
        
        featRow = np.mean(MFCC_sequence, axis=1)
        features[targIter, :] = featRow
        target[targIter] = instLabel
        targIter += 1

    instLabel += 1

    
soundDF = pd.DataFrame(features)
soundDF['Target'] = target

soundDF.to_csv(r'/Users/mosesmakangila/Documents/ML INDEEPENDENT STUDY/MFCC_vowelclean_features.csv', index=False, header=True)