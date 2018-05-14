#!/usr/bin/env python3
"""
compute_image_vectors.py - computes image vectors of jpegs

Usage:
  compute_image_vectors.py <image_dir> <output_prefix>
"""
import glob
import os
import pathlib

from docopt import docopt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

MODULE_NAME = 'nasnet_large'
MODULE_URL = 'https://tfhub.dev/google/imagenet/' + MODULE_NAME + '/feature_vector/1'


def create_module_graph(module_spec):
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_image_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_image_tensor)
    return graph, bottleneck_tensor, resized_image_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_image_tensor,
                            bottleneck_tensor):
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def add_jpeg_decoding(module_spec):
    height, width = hub.get_expected_image_size(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([height, width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    return jpeg_data, resized_image


def main():
    args = docopt(__doc__, help=True)
    image_dir = args['<image_dir>']
    output_prefix = args['<output_prefix>']
    image_paths = glob.glob(image_dir + '/*.jpg')
    if len(image_paths) == 0:
        print('no images (or image directory) found.', file=sys.stderr)
        sys.exit(1)

    module_spec = hub.load_module_spec(MODULE_URL)
    graph, bottleneck_tensor, resized_image_tensor = create_module_graph(module_spec)

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        output_dir = os.path.join(output_prefix, 'feature_vectors', MODULE_NAME)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            fn = os.path.basename(img_path)
            assert fn.endswith('.jpg')
            fn = fn[:-4] + '.npy'
            output_path = os.path.join(output_dir, fn)
            if not os.path.exists(output_path):
                image_data = open(img_path, 'rb').read()
                bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                                                            resized_image_tensor, bottleneck_tensor)
                np.save(output_path, bottleneck_values)


if __name__ == '__main__':
    main()
