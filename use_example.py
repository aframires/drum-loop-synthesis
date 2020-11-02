import os
import numpy as np
import argparse
from scipy import signal
import math

from model import Model


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

def main(config):
    model = Model(config)
    gen_pattern = [[1,0,0,0,0,0.7,0,0,0,0,0,0], #hh
     [0,0,1,0,1,1,0,0,0.2,0,1,0], #snare
     [1,1,1,1,1,1,1,1,1,1,1,1]]

    features_hh = [0.5,0.5,0.5,0.5,0.5,0.9,0.9]
    features_kick = [0.5,0.1,0.2,0.8,0.5,0.5,0.5]
    features_snare = [0.5,0.5,0.5,0.5,1,0.5,0.5]
    hpcp = [0,0,0,0,0,1,1,0,1,0,0,0]
    pattern = generate_gaussians(gen_pattern)
    pattern = np.expand_dims(np.array(pattern).T,0)

    model.use_model(pattern, hpcp, features_kick, features_hh, features_kick)
    
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--model', type=str, default="multi_noenv", help='Models to use, must be in multi_env, multi, wavespec, wav or spec')

    parser.add_argument('--log_dir', type=str, default="/home/pc2752/share/loop_synth/", help='The directory where the models are saved')

    parser.add_argument('--val_file', type=str, default="/home/pc2752/share/loop_synth/feats/loop_feats_val.hdf5", help='Path to the file containing validation features')

    parser.add_argument('--output_dir', type=str, default="/home/pc2752/share/drum-loop-synthesis/outputs/", help='Directory to save the outputs in')

    config = parser.parse_args()

    assert config.model in ["multi_noenv", "multi", "wavspec", "wav", "spec"]

    if config.model == 'multi_noenv':
        config.log_dir = os.path.join(config.log_dir,'log_multi_noenv/')
        config.output_features = 1
        config.rhyfeats = 3
        config.encoder_layers = 10
        config.sample_len = 29538
    if config.model == 'multi':
        config.log_dir = os.path.join(config.log_dir,'log_multi/')
        config.output_features = 1
        config.rhyfeats = 4
        config.encoder_layers = 10
        config.sample_len = 29538
    if config.model == 'wavspec':
        config.log_dir = os.path.join(config.log_dir,'log_wav/')
        config.output_features = 1
        config.rhyfeats = 4
        config.encoder_layers = 10
        config.sample_len = 29538
    if config.model == 'wav':
        config.log_dir = os.path.join(config.log_dir,'log_wavonly/')
        config.output_features = 1
        config.rhyfeats = 4
        config.encoder_layers = 10
        config.sample_len = 29538
    if config.model == 'spec':
        config.log_dir = os.path.join(config.log_dir,'log_stft_old/')
        config.output_features = 513
        config.rhyfeats = 4
        config.encoder_layers = 6
        config.sample_len = 159


    config.num_epochs = 2500
    config.batch_size = 16
    config.filter_len = 5
    config.filters = 32
    config.fs = 16000
    
    config.max_phr_len = 159
    config.n_fft = 1024
    config.hop_size = 180
    config.input_features = 31

    config.kernel_size = 2
    config.num_filters = 100
    config.skip_filters = 240
    config.first_conv = 10
    config.dilation_rates = [1,2,4,1,2,4,1,2,4,1,2,4]

    main(config)

