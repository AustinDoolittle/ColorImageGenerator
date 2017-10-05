import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import os
import argparse as ap
import numpy as np
from PIL import Image
import h5py

def parse_args(args):
    parser = ap.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, help='The size of the batch')
    parser.add_argument('--dimensions', type=int, nargs=2, default=[384, 384], help='The size of the images fed in as input')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to train on ')
    return parser.parse_args(args)

def load_dataset(filename, split=None):
    if split is None:
        split = (.8, .2)

    try:
        f = h5py.File(filename, 'r+')
    except IOError as ioe:
        print 'ERROR There was an error while loading the h5py file: ' + str(ioe)
        sys.exit()
    arr = np.array(f['images'])
    r, g, b = arr[:, :, :, 0], arr[:, :, :, 1], arr[:, :, :, 2]
    gray_arr = np.expand_dims(0.2989 * r + 0.5870 * g + 0.1140 * b, axis=3)
    arr_split = int(split[0] * arr.shape[0])
    return gray_arr[:arr_split, :, :, :], arr[:arr_split, :, :, :], gray_arr[arr_split:, :, :, :], arr[arr_split:, :, :,
                                                                                                   :]

def graph(input):
    conv1 = slim.conv2d(input, 32, [5, 5], stride=2, scope='conv1')
    conv2 = slim.conv2d(conv1, 64, [3, 3], stride=2, scope='conv2')
    conv3 = slim.conv2d(conv2, 128, [3, 3], stride=2, scope='conv3')
    conv4 = slim.conv2d(conv3, 256, [1, 1], stride=2, scope='conv4')
    deconv1 = slim.conv2d_transpose(conv4, 128, [3, 3], stride=2, scope='deconv1')
    deconv2 = slim.conv2d_transpose(deconv1, 64, [3, 3], stride=2, scope='deconv2')
    deconv3 = slim.conv2d_transpose(deconv2, 32, [5, 5], stride=2, scope='deconv3')
    deconv4 = slim.conv2d_transpose(deconv3, 3, [5, 5], stride=2, scope='deconv4')

    return deconv4

def main(argv):

    args = parse_args(argv)

    input_x = tf.placeholder(tf.float32, [None, 384, 384, 1])
    output_y = tf.placeholder(tf.float32, [None, 384, 384, 3])

    output_y_ = graph(input_x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_y, logits=output_y_))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_x, train_y, test_x, test_y = load_dataset('/Users/austindoolittle/Downloads/food-images-food-101/food_c101_n1000_r384x384x3.h5')

    test_index = 0
    for i in xrange(args.epochs):
        np.random.shuffle(train_x)
        last_j = 0
        for j in xrange(0, train_x.shape[0], args.batch_size):
            x = np.expand_dims(train_x[last_j:j, :, :, :], axis=0)
            y = np.expand_dims(train_y[last_j:j, :, :, :], axis=0)
            sess.run(updates, {input_x: x, output_y: y})
            last_j = j

        print 'Epoch ' + str(i) + '/' + str(args.epochs)

        print 'Writing example to file'
        actual_index = test_index % test_x.shape[0]
        test_index += 1
        x = np.expand_dims(test_x[actual_index, :, :, :], axis=0)
        out = sess.run(output_y_, {input_x: x})
        img = Image.fromarray(out[0, :, :, :], 'RGB')
        if not os.path.exists('./Images'):
            os.makedirs('./Images')

        img.save('./Images/img' + str(test_index) + '.png')


if __name__ == '__main__':
    main(sys.argv[1:])
