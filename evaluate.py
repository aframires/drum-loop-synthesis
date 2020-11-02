import os

import argparse
from model import Model

def main(config):
    model = Model(config)
    model.test_model()
    command = "ls --color=never -d {} > {}".format(os.path.join(config.output_dir, 'gt_*'), os.path.join(config.output_dir, 'gt.cvs'))
    os.system(command)
    command = "ls --color=never -d {} > {}".format(os.path.join(config.output_dir, 'output_*_{}.wav'.format(config.model)), \
        os.path.join(config.output_dir, 'output_{}.cvs'.format(config.model)))
    os.system(command)

    command = "python -m FAD_evaluation.frechet_audio_distance.create_embeddings_main --input_files {} --stats {}".format(os.path.join(config.output_dir, 'gt.cvs'),\
        os.path.join(config.output_dir, 'gt_stats'))
    os.system(command)

    command = "python -m FAD_evaluation.frechet_audio_distance.create_embeddings_main --input_files {} --stats {}".format(os.path.join(config.output_dir,\
     'output_{}.cvs'.format(config.model)),os.path.join(config.output_dir, 'gt_stats'))
    os.system(command)
        
    command = "python -m FAD_evaluation.frechet_audio_distance.create_embeddings_main --input_files {} --stats {}".format(os.path.join(config.output_dir, 'gt.cvs'),\
        os.path.join(config.output_dir, '{}_stats'.format(config.model)))
    os.system(command)

    command = "python -m frechet_audio_distance.compute_fad --background_stats {} --test_stats {}".format(os.path.join(config.output_dir, 'gt_stats'),\
     os.path.join(config.output_dir, '{}_stats'.format(config.model)))
    os.system(command)

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