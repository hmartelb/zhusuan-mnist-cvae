
from __future__ import absolute_import, division, print_function

import argparse
import gzip
import os
import time

import numpy as np
import six
import tensorflow as tf
import zhusuan as zs
from six.moves import cPickle as pickle

import dataset
import utils


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(y, x_dim, z_dim, n):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1)

    # Concatenate z and y
    z = tf.concat(axis=1, values=[z,y])

    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)

    x_mean = bn.deterministic("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1, dtype=tf.float32)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, y, z_dim):
    bn = zs.BayesianNet()
    
    # Concatenate x and y
    x = tf.concat(axis=1, values=[x,y])

    h = tf.layers.dense(x, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)

    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1)
    return bn


def train_vae(args):
    # Load MNIST
    data_path = os.path.join(args.data_dir, "mnist.pkl.gz")
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset.load_mnist_realval(data_path)
    x_train = np.random.binomial(1, x_train, size=x_train.shape)
    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]

    # Define model parameters
    z_dim = args.z_dim

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    n = tf.placeholder(tf.int32, shape=[], name="n")

    # Get the models
    model = build_gen(y, x_dim, z_dim, n)
    variational = build_q_net(x, y, z_dim)

    # Calculate ELBO
    lower_bound = zs.variational.elbo(model, {"x": x }, variational=variational)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    infer_op = optimizer.minimize(cost)

    # Random generation
    x_gen = tf.reshape(model.observe()["x_mean"], [-1, 28, 28, 1])

    # Compute class labels
    labels = []
    for c in range(10):
        l = np.zeros((100, 10))
        l[:,c] = 1
        labels.append(l)

    epochs = args.epochs
    batch_size = args.batch_size
    iters = x_train.shape[0] // batch_size

    saver = tf.train.Saver(max_to_keep=10)
    save_model_freq = min(100, args.epochs)

    # Run the Inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt_file = tf.train.latest_checkpoint(args.checkpoints_path)
        begin_epoch = 1

        if(ckpt_file is not None):
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(1, epochs+1):
            time_epoch = -time.time()
            lbs = []
            for t in range(iters):
                x_batch = x_train[t*batch_size:(t+1)*batch_size]
                y_batch = y_train[t*batch_size:(t+1)*batch_size]

                _, lb = sess.run(
                    [infer_op, lower_bound],
                    feed_dict={
                        x: x_batch,
                        y: y_batch,
                        n: batch_size
                    }
                )
                lbs.append(lb)
            
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(epoch, time_epoch, np.mean(lbs)))

            if(epoch % args.save_model_freq == 0):
                save_path = os.path.join(args.checkpoints_path, "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
            
            if epoch % args.save_img_freq == 0:
                for c in range(10):
                    images = sess.run(x_gen, feed_dict={y: labels[c], n: 100 })
                    name = os.path.join(args.results_path, str(epoch).zfill(3), "{}.png".format(c))
                    utils.save_image_collections(images, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hardware setup
    parser.add_argument('--gpu', default='0')

    # Variational parameters
    parser.add_argument('--z_dim', default=40, type=float)
    
    # Training parameters
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    # Saving parameters
    parser.add_argument('--save_model_freq', default=10, type=int)
    parser.add_argument('--save_img_freq', default=10, type=int)
    
    # Paths
    parser.add_argument('--data_dir', default=os.path.join('data'))
    parser.add_argument('--results_path', default=os.path.join('results'))
    parser.add_argument('--checkpoints_path', default=os.path.join('checkpoints'))
    
    args = parser.parse_args()

    # Select which GPU to use and enable mixed precision
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    train_vae(args)
