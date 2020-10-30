import sys
import os,re
import collections
import csv
import soundfile as sf
import numpy as np
from scipy.stats import norm
# import pyworld as pw
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm




def griffinlim(spectrogram, config, n_iter = 50, window = 'hann', verbose = False):
    n_fft = config.n_fft
    hop_length = config.hop_size


    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        inverse = istft(spectrogram,angles, hopsize=hop_length, nfft=n_fft, fs=config.fs)
        rebuilt = stft(inverse, hopsize=hop_length, nfft=n_fft, fs=config.fs)[:spectrogram.shape[0],:]
        angles = np.exp(1j * np.angle(rebuilt))
        progress(i,n_iter)

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))
    inverse = istft(spectrogram, angles)

    return inverse


def shuffle_two(a,b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)
    return a2, b2


def stft(data, window=np.hanning(1024),
         hopsize=180, nfft=1024.0, fs=16000.0):
    """
    X, F, N = stft(data,window=sinebell(2048),hopsize=1024.0,
                   nfft=2048.0,fs=44100)
                   
    Computes the short time Fourier transform (STFT) of data.
    
    Inputs:
        data                  :
            one-dimensional time-series to be analyzed
        window=sinebell(2048) :
            analysis window
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
        fs=44100.0            :
            sampling rate of the signal
        
    Outputs:
        X                     :
            STFT of data
        F                     :
            values of frequencies at each Fourier bins
        N                     :
            central time at the middle of each analysis
            window
    """
    
    # window defines the size of the analysis windows
    lengthWindow = window.size
    
    lengthData = data.size
    
    # should be the number of frames by YAAFE:
    numberFrames = np.ceil(lengthData / np.double(hopsize)) + 2
    # to ensure that the data array s big enough,
    # assuming the first frame is centered on first sample:
    newLengthData = (numberFrames-1) * hopsize + lengthWindow

    # import pdb;pdb.set_trace()
    
    # !!! adding zeros to the beginning of data, such that the first window is
    # centered on the first sample of data

    # import pdb;pdb.set_trace()
    if len(data.shape)>1:
        data = np.mean(data, axis = -1)
    data = np.concatenate((np.zeros(int(lengthWindow/2)), data))
    
    # zero-padding data such that it holds an exact number of frames

    data = np.concatenate((data, np.zeros(int(newLengthData - data.size))))
    
    # the output STFT has nfft/2+1 rows. Note that nfft has to be an even
    # number (and a power of 2 for the fft to be fast)
    numberFrequencies = nfft / 2 + 1
    
    STFT = np.zeros([int(numberFrames), int(numberFrequencies)], dtype=complex)
    
    # storing FT of each frame in STFT:
    for n in np.arange(numberFrames):
        beginFrame = n*hopsize
        endFrame = beginFrame+lengthWindow
        frameToProcess = window*data[int(beginFrame):int(endFrame)]
        STFT[int(n),:] = np.fft.rfft(frameToProcess, np.int32(nfft), norm="ortho")
        
    # frequency and time stamps:
    F = np.arange(numberFrequencies)/np.double(nfft)*fs
    N = np.arange(numberFrames)*hopsize/np.double(fs)
    
    return STFT

def istft(mag, phase, window=np.hanning(1024),
         hopsize=180, nfft=1024.0, fs=16000.0,
          analysisWindow=None):
    """
    data = istft_norm(X,window=sinebell(2048),hopsize=1024.0,nfft=2048.0,fs=44100)
    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.
    Inputs:
        X                     :
            STFT of the signal, to be \"inverted\"
        window=sinebell(2048) :
            synthesis window
            (should be the \"complementary\" window
            for the analysis window)
        hopsize=1024.0        :
            hopsize for the analysis
        nfft=2048.0           :
            number of points for the Fourier computation
            (the user has to provide an even number)
    Outputs:
        data                  :
            time series corresponding to the given STFT
            the first half-window is removed, complying
            with the STFT computation given in the
            function stft
    """
    X = mag * np.exp(1j*phase)
    X = X.T
    if analysisWindow is None:
        analysisWindow = window

    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = X.shape
    lengthData = int(hopsize*(numberFrames-1) + lengthWindow)

    normalisationSeq = np.zeros(lengthData)

    data = np.zeros(lengthData)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], np.int32(nfft), norm = 'ortho')
        frameTMP = frameTMP[:lengthWindow]
        normalisationSeq[beginFrame:endFrame] = (
            normalisationSeq[beginFrame:endFrame] +
            window * analysisWindow)
        data[beginFrame:endFrame] = (
            data[beginFrame:endFrame] + window * frameTMP)

    data = data[int(lengthWindow/2.0):]
    normalisationSeq = normalisationSeq[int(lengthWindow/2.0):]
    normalisationSeq[normalisationSeq==0] = 1.

    data = data / normalisationSeq

    return data

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()
