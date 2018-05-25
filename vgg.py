import os.path as path
import urllib.request
import shutil

import numpy as np
import scipy.io
import tensorflow as tf

VGG_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'


def net(input_image, output_layer, is_style):
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
              'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
              'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
              'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
              'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')
    mean_pixel = np.array([123.68, 116.779, 103.939])

    assert output_layer in layers
    weights = _get_vgg_weights()
    current = input_image - mean_pixel
    for i, layer in enumerate(layers):
        kind = layer[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)

        if layer == output_layer:
            if is_style:
                flat = tf.reshape(current, (-1, current.shape[3]))
                num_feats = tf.shape(flat)[1]
                current = tf.matmul(flat, flat, transpose_a=True) / tf.cast(num_feats, flat.dtype)

            output = tf.reshape(current, [tf.size(current)])
            return output


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')


def _get_vgg_weights():
    vgg_path = VGG_URL.split('/')[-1]
    if not path.exists(vgg_path):
        with urllib.request.urlopen(VGG_URL) as response:
            if response.code == 200:
                with open(vgg_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            else:
                raise Exception('could not download VGG weights')

    data = scipy.io.loadmat(vgg_path)
    weights = data['layers'][0]
    return weights
