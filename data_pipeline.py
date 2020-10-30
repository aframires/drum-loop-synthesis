import numpy as np
import os
import time
import h5py
import random
import matplotlib.pyplot as plt
import collections
import utils




def data_gen(config):

    hdf5_file = h5py.File(config.val_file, mode='r')
    audios = hdf5_file["waveform"]
    if config.model=="spec":
        act = hdf5_file["new_act"]
    else:
        act = hdf5_file["act"]
    features = hdf5_file["features"]

    max_feats = [  1.       ,   1.       ,   1.       ,   1.       ,   1.       ,
         1.       ,   1.       ,   1.       ,   1.       ,   1.       ,
         1.       ,   1.       ,  69.57681  ,  67.66642  ,  80.19115  ,
        71.689445 ,  61.422714 , 100.       ,  71.406494 ,  32.789112 ,
         1.       ,  85.24432  ,  67.71172  ,   2.491137 ,   0.5797179,
        87.83693  ,  69.738235 ,  71.95989  ,  82.336105 ,  75.53646  ,
        71.00043  , 100.       ,  81.7323   ]

    in_indecis = np.arange(len(features))

    act_sum = act[()].sum(axis=1).sum(axis=1)

    remove_indecis = np.argwhere(act_sum==0)

    
    in_indecis = np.array([x for x in np.arange(len(features)) if x  not in remove_indecis])


    for i, idx_batch in enumerate(np.arange(int(len(in_indecis)/config.batch_size))):

        i_start = i * config.batch_size
        i_end = min([(i + 1) * config.batch_size, len(in_indecis)])
        indecis = in_indecis[i_start:i_end]
        indecis = [x for x in indecis]


        out_audios = audios[indecis]
        out_act = act[indecis]
        out_act+=1e-15
        out_act = out_act/out_act.max(axis=-2)[:,np.newaxis,:]

        out_features = features[indecis]/max_feats


        out_features = np.concatenate((out_features[:,:19], out_features[:,21:]), axis = 1)


        yield np.expand_dims(out_audios, -1), out_act, out_features, int(len(in_indecis)/config.batch_size)
