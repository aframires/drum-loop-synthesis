import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf

import utils
import modules_tf as modules
from data_pipeline import data_gen

class Model(object):

    def __init__(self, config):
        self.config = config

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.sample_len, config.rhyfeats),name='input_placeholder')

        self.cond_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.input_features),name='cond_placeholder')

        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, 29538, 1),name='output_placeholder')       

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        with tf.variable_scope('Perc_Synth') as scope:
            self.output_wav = modules.full_network(self.cond_placeholder,self.input_placeholder,  self.is_train, self.config)

    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver()


        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(self.config.log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def test_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = self.config.log_dir)
        val_generator = data_gen(self.config)
        count_batch = 0
        for batch_count, [out_audios, out_envelopes, out_features, total_count] in enumerate(val_generator):
            feed_dict = {self.input_placeholder: out_envelopes[:,:,:self.config.rhyfeats], self.cond_placeholder: out_features,\
             self.output_placeholder: out_audios, self.is_train: False}
            output_full = sess.run(self.output_wav, feed_dict=feed_dict)

            for count in range(self.config.batch_size):
                if self.config.model == "spec":
                    out_audio = utils.griffinlim(np.exp(output_full[count]) -1, self.config)
                else:
                    out_audio = output_full[count]
                output_file = os.path.join(self.config.output_dir,'output_{}_{}_{}.wav'.format(batch_count, count, self.config.model))
                sf.write(output_file, np.clip(out_audio,-1,1), self.config.fs)
                sf.write(os.path.join(self.config.output_dir,'gt_{}_{}.wav'.format(batch_count, count)), out_audios[count],self.config.fs)
            utils.progress(batch_count, total_count)


    def mix_model(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = self.config.log_dir)
        val_generator = data_gen(self.config)
        count_batch = 0
        for batch_count, [out_audios, out_envelopes, out_features, total_count] in enumerate(val_generator):
            out_features_copy = np.copy(out_features)
            for j in range(int(len(out_features)/2)-1):
                out_features[j] = out_features_copy[-1-j]
                out_features[-1-j] = out_features_copy[j]

            feed_dict = {self.input_placeholder: out_envelopes[:,:,:self.config.rhyfeats], self.cond_placeholder: out_features,\
             self.output_placeholder: out_audios, self.is_train: False}
            output_full = sess.run(self.output_wav, feed_dict=feed_dict)

            for count in range(self.config.batch_size):
                if self.config.model == "spec":
                    out_audio = utils.griffinlim(np.exp(output_full[count]) -1, self.config)
                else:
                    out_audio = output_full[count]
                output_file = os.path.join(self.config.output_dir,'output_{}_{}_{}.wav'.format(batch_count, count, self.config.model))
                sf.write(output_file, np.clip(out_audio,-1,1), self.config.fs)
                sf.write(os.path.join(self.config.output_dir,'gt_{}_{}.wav'.format(batch_count, count)), out_audios[count],self.config.fs)
            utils.progress(batch_count, total_count)

    def source_separate(self):
        sess = tf.Session()
        self.load_model(sess, log_dir = self.config.log_dir)
        val_generator = data_gen(self.config)
        count_batch = 0
        for batch_count, [out_audios, out_envelopes, out_features, total_count] in enumerate(val_generator):
            out_envelopes_bass = np.copy(out_envelopes)
            out_envelopes_bass[:,:,1:3] = 0
            out_envelopes_mid = np.copy(out_envelopes)
            out_envelopes_mid[:,:,0] = 0
            out_envelopes_mid[:,:,2] = 0
            out_envelopes_high = np.copy(out_envelopes)
            out_envelopes_high[:,:,:2] = 0

            feed_dict = {self.input_placeholder: out_envelopes_bass[:,:,:self.config.rhyfeats], self.cond_placeholder: out_features,\
             self.output_placeholder: out_audios, self.is_train: False}
            output_bass = sess.run(self.output_wav, feed_dict=feed_dict)

            feed_dict = {self.input_placeholder: out_envelopes_mid[:,:,:self.config.rhyfeats], self.cond_placeholder: out_features,\
             self.output_placeholder: out_audios, self.is_train: False}
            output_mid = sess.run(self.output_wav, feed_dict=feed_dict)

            feed_dict = {self.input_placeholder: out_envelopes_high[:,:,:self.config.rhyfeats], self.cond_placeholder: out_features,\
             self.output_placeholder: out_audios, self.is_train: False}
            output_high = sess.run(self.output_wav, feed_dict=feed_dict)

            for count in range(self.config.batch_size):
                if self.config.model == "spec":
                    out_audio_bass = utils.griffinlim(np.exp(output_bass[count]) -1, self.config)
                    out_audio_mid = utils.griffinlim(np.exp(output_mid[count]) -1, self.config)
                    out_audio_high = utils.griffinlim(np.exp(output_high[count]) -1, self.config)
                else:
                    out_audio_bass = output_bass[count]
                    out_audio_mid = output_mid[count]
                    out_audio_high = output_high[count]

                output_file_bass = os.path.join(self.config.output_dir,'output_{}_{}_{}_bass.wav'.format(batch_count, count, self.config.model))
                sf.write(output_file_bass, np.clip(out_audio_bass,-1,1), self.config.fs)
                output_file_mid = os.path.join(self.config.output_dir,'output_{}_{}_{}_mid.wav'.format(batch_count, count, self.config.model))
                sf.write(output_file_mid, np.clip(out_audio_mid,-1,1), self.config.fs)
                output_file_high = os.path.join(self.config.output_dir,'output_{}_{}_{}_high.wav'.format(batch_count, count, self.config.model))
                sf.write(output_file_high, np.clip(out_audio_high,-1,1), self.config.fs)

                sf.write(os.path.join(self.config.output_dir,'gt_{}_{}.wav'.format(batch_count, count)), out_audios[count],self.config.fs)
            utils.progress(batch_count, total_count)


    def use_model(self, pattern, hpcp, features_kick, features_snare, features_hh):
        sess = tf.Session()
        self.load_model(sess, log_dir = self.config.log_dir)
        pattern = np.repeat(pattern, self.config.batch_size, 0)
        features = np.concatenate((np.array(hpcp), np.array(features_kick), np.array(features_snare), np.array(features_hh))) 
        features = np.repeat(np.expand_dims(features, 0), self.config.batch_size, 0)
        features = np.concatenate((features[:,:19], features[:,21:]), axis = 1)
        feed_dict = {self.input_placeholder: pattern, self.cond_placeholder: features, self.is_train: False}
        output_full = sess.run(self.output_wav, feed_dict=feed_dict)
        output_file = os.path.join(self.config.output_dir,'output_{}.wav'.format(self.config.model))
        sf.write(output_file, np.clip(output_full[0],-1,1), self.config.fs)

