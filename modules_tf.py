import tensorflow as tf
import numpy as np



def encoder_conv_block(inputs, layer_num, is_train, num_filters, filter_len):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block(inputs, layer, layer_num, is_train, num_filters, filter_len):

    deconv = tf.image.resize_images(inputs, size=(layer.shape[1],1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num))), training = is_train, name =  "DBN_"+str(layer_num))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv



def encoder_decoder_archi(inputs, is_train, config):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block(encoded, i, is_train, config.filters, config.filter_len)

        encoder_layers.append(encoded)
    
    encoder_layers.reverse()

    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):

        decoded = decoder_conv_block(decoded, encoder_layers[i+1], i, is_train, config.filters, config.filter_len)
    return decoded


def full_network(condsi, env, is_train, config):

    conds = tf.tile(tf.reshape(condsi,[-1,1,config.input_features]),[1,config.sample_len,1])

    inputs = tf.concat([conds, env], axis = -1)

    inputs = tf.reshape(inputs, [-1, config.sample_len , 1, config.input_features+config.rhyfeats])

    inputs = tf.nn.relu(tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in"), training = is_train), name = "S_in_BN")

    output = encoder_decoder_archi(inputs, is_train, config)

    output = tf.layers.batch_normalization(tf.layers.dense(output, config.output_features, name = "Fu_F"), training = is_train)

    output = tf.reshape(output, [-1, config.sample_len, config.output_features])

    return output
