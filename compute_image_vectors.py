#!/usr/bin/env python3
"""
compute_image_vectors.py - computes image vectors of jpegs

Usage:
  compute_image_vectors.py <image_dir> <output_prefix> <module_name>
"""
import glob
import os
from os import path
import pathlib
import sys

from docopt import docopt
import numpy as np


def create_vgg_graph(layer, is_style):
    height, width = (256, 256)
    with tf.Graph().as_default() as graph:
        resized_image_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        bottleneck_tensor = vgg.net(resized_image_tensor, layer, is_style)
    return height, width, graph, bottleneck_tensor, resized_image_tensor


def create_module_graph(module_spec):
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_image_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_image_tensor)
    return height, width, graph, bottleneck_tensor, resized_image_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor,
                            resized_image_tensor, bottleneck_tensor):
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    bottleneck_values = sess.run(bottleneck_tensor, {resized_image_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def add_jpeg_decoding(height, width):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([height, width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    return jpeg_data, resized_image


def process_module_name(module_name):
    is_vgg, is_style, layer = False, False, None
    if module_name.startswith('vgg-'):
        is_vgg = True
        layer = module_name[4:]
        if module_name.endswith('-style'):
            is_style = True
            layer = layer[:-6]
    return is_vgg, is_style, layer


def main():
    args = docopt(__doc__, help=True)
    image_dir = args['<image_dir>']
    image_paths = glob.glob(image_dir + '/*.jpg')
    if len(image_paths) == 0:
        print('no images (or image directory) found.', file=sys.stderr)
        sys.exit(1)
    output_prefix = args['<output_prefix>']
    module_name = args['<module_name>']
    is_vgg, is_style, layer = process_module_name(module_name)

    global tf, hub
    import tensorflow as tf
    import tensorflow_hub as hub

    if is_vgg:
        global vgg
        import vgg
        height, width, graph, bottleneck_tensor, resized_image_tensor = create_vgg_graph(
            layer, is_style)

    else:
        module_url = 'https://tfhub.dev/google/imagenet/' + module_name + '/feature_vector/1'
        module_spec = hub.load_module_spec(module_url)
        height, width, graph, bottleneck_tensor, resized_image_tensor = create_module_graph(
            module_spec)

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(height, width)

        output_dir = path.join(output_prefix, 'feature_vectors', module_name)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        all = []
        for img_path in image_paths:
            fn = path.basename(img_path)
            assert fn.endswith('.jpg')
            fn = fn[:-4] + '.npy'
            output_path = path.join(output_dir, fn)
            if not path.exists(output_path):
                image_data = open(img_path, 'rb').read()
                bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor,
                                                            decoded_image_tensor,
                                                            resized_image_tensor, bottleneck_tensor)
                np.save(output_path, bottleneck_values)
                all.append(bottleneck_values.reshape(1, bottleneck_values.size))

        np.save(path.join(output_path, 'all.npy'), np.concatenate(all))


if __name__ == '__main__':
    main()
