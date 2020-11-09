from ADTLib import ADT
import essentia.standard as es
import essentia as e
import numpy as np
import soundfile as sf
import timbral_models
from scipy import signal
import math

import ntpath
import os

def file_to_hpcp(loop):
    loop = e.array(loop)

    windowing = es.Windowing(type='blackmanharris62')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='magnitude',
                                    magnitudeThreshold=0.001,
                                    maxPeaks=20,
                                    minFrequency=20,
                                    maxFrequency=8000)
    hpcp = es.HPCP(maxFrequency=8000)
    spec_group = []
    hpcp_group = []
    for frame in es.FrameGenerator(loop,frameSize=1024,hopSize=512):
        windowed = windowing(frame)
        fft = spectrum(windowed)
        frequencies, magnitudes = spectral_peaks(fft)
        final_hpcp = hpcp(frequencies, magnitudes)
        spec_group.append(fft)
        hpcp_group.append(final_hpcp)

    mean_hpcp = np.mean(np.array(hpcp_group).T, axis = 1)
    #normalize to 1
    mean_hpcp = mean_hpcp/mean_hpcp.max()


    return mean_hpcp

def analysis_function(loop,sampleRate):
    lp_filter = es.LowPass(cutoffFrequency=90,sampleRate=sampleRate)
    bp_filter = es.BandPass(bandwidth=100 ,cutoffFrequency=280,sampleRate=sampleRate)
    hp_filter = es.HighPass(cutoffFrequency=9000,sampleRate=sampleRate)

    [_, pattern] = ADT([loop], output_act='yes', tab='no', save_dir="analysis/")

    audio_file=es.MonoLoader(filename=loop,sampleRate=sampleRate)


    loop_basename = ntpath.basename(loop)
    lpf_audio = lp_filter(audio_file())
    bpf_audio = bp_filter(audio_file())
    hpf_audio = hp_filter(audio_file())

    sf.write("analysis/lpf_" + loop_basename, lpf_audio, sampleRate)
    sf.write("analysis/bpf_" + loop_basename, bpf_audio, sampleRate)
    sf.write("analysis/hpf_" + loop_basename, hpf_audio, sampleRate)

    features_kick = timbral_models.timbral_extractor("analysis/lpf_" + loop_basename, clip_output=True)
    features_snare = timbral_models.timbral_extractor("analysis/bpf_" + loop_basename, clip_output=True)
    features_hh = timbral_models.timbral_extractor("analysis/hpf_" + loop_basename, clip_output=True)

    hpcp = file_to_hpcp(audio_file())

    return pattern,hpcp,features_kick,features_snare,features_hh

def generate_gaussians(pattern):
    bar_len = 29538
    gauss_std =100
    gauss_window = 1001
    gauss = signal.gaussian(gauss_window,gauss_std)
    gauss_patterns = []
    for inst_pattern in pattern:
        gauss_pat = np.zeros(bar_len)
        for idx, val in enumerate(inst_pattern):
            if val != 0:
                center_pos = math.floor(idx * bar_len/16)
                if idx != 0:
                    left_pos = center_pos - math.ceil(gauss_window)
                    gauss_pat[left_pos:left_pos + gauss_window] = val * gauss
                else:
                    gauss_pat[0:math.ceil(gauss_window/2)] = val * gauss[math.floor(gauss_window/2):]
        gauss_patterns.append(gauss_pat)
    return gauss_patterns

class Config:
    def __init__(self,selected_model):
        self.model = selected_model
        self.log_dir = "/content/drum-loop-synthesis/models/"
        self.val_file = ""
        self.output_dir = "/content/drum-loop-synthesis/output/" 
        if self.model == 'multi_noenv':
            self.log_dir = os.path.join(self.log_dir,'log_multi_noenv/')
            self.output_features = 1
            self.rhyfeats = 3
            self.encoder_layers = 10
            self.sample_len = 29538
        if self.model == 'multi':
            self.log_dir = os.path.join(self.log_dir,'log_multi/')
            self.output_features = 1
            self.rhyfeats = 4
            self.encoder_layers = 10
            self.sample_len = 29538
        if self.model == 'wavspec':
            self.log_dir = os.path.join(self.log_dir,'log_wav/')
            self.output_features = 1
            self.rhyfeats = 4
            self.encoder_layers = 10
            self.sample_len = 29538
        if self.model == 'wav':
            self.log_dir = os.path.join(self.log_dir,'log_wavonly/')
            self.output_features = 1
            self.rhyfeats = 4
            self.encoder_layers = 10
            self.sample_len = 29538
        if self.model == 'spec':
            self.log_dir = os.path.join(self.log_dir,'log_stft_old/')
            self.output_features = 513
            self.rhyfeats = 4
            self.encoder_layers = 6
            self.sample_len = 159


        self.num_epochs = 2500
        self.batch_size = 16
        self.filter_len = 5
        self.filters = 32
        self.fs = 16000
        
        self.max_phr_len = 159
        self.n_fft = 1024
        self.hop_size = 180
        self.input_features = 31

        self.kernel_size = 2
        self.num_filters = 100
        self.skip_filters = 240
        self.first_conv = 10
        self.dilation_rates = [1,2,4,1,2,4,1,2,4,1,2,4]



def generate(selected_model, pattern, hpcp, features_kick, features_snare, features_hh):
    from model import Model
    
    config = Config(selected_model)
    model = Model(config)
    model.use_model(pattern, hpcp, features_kick, features_snare, features_kick)