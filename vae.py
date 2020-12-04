from __future__ import absolute_import, division, print_function

import argparse
import os
import time

import numpy as np
import tensorflow as tf
import zhusuan as zs

import dataset
import model
import utils

def train_vae(args):
    # Load MNIST dataset
    data_path = os.path.join(args.data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = dataset.load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    n_iter = x_train.shape[0] // args.batch_size
    
    # Define model parameters x and z
    x_dim = x_train.shape[1]
    z_dim = args.z_dim
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")

    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input), tf.int32)
    n = tf.placeholder(tf.int32, shape=[], name="n")
    std_noise = tf.placeholder_with_default(0., shape=[], name="std_noise")
 
    # Get the models
    gen_model = model.build_gen(x_dim, z_dim, n, n_particles)
    q_model = model.build_q_net(x, z_dim, n_particles, std_noise)
    variational = q_model.observe()
 
    # Calculate ELBO
    lower_bound = zs.variational.elbo(
        gen_model, {"x": x}, 
        variational=variational, 
        axis=0
    )
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    infer_op = optimizer.minimize(cost)

    saver = tf.train.Saver(max_to_keep=10)
    save_model_freq = min(100, args.epochs)
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        ckpt_file = tf.train.latest_checkpoint(args.checkpoints_path)
        begin_epoch = 1

        if(ckpt_file is not None):
            print('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, args.epochs + 1):
            time_epoch = - time.time()
            np.random.shuffle(x_train)
            
            lbs = []
            for t in range(n_iter):
                dataset.show_progress(t, 1, n_iter)
                x_batch = x_train[t * args.batch_size:(t + 1) * args.batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                    feed_dict={
                        x_input: x_batch,
                        n_particles: 1,
                        n: args.batch_size
                    })
                lbs.append(lb)
 
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(epoch, time_epoch, np.mean(lbs)))

            if(epoch % save_model_freq == 0):
                print('Saving model...')
                
                save_path = os.path.join(args.checkpoints_path, "vae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                
                saver.save(sess, save_path)
                print('Done')
 
        x_gen = tf.reshape(gen_model.observe()["x_mean"], [-1, 28, 28, 1])
        images = sess.run(x_gen, feed_dict={n: 100, n_particles: 1})
        name = os.path.join(args.result_path, "random_samples.png")
        utils.save_image_collections(images, name)
 
        test_n = [3, 2, 1, 90, 95, 23, 11, 0, 84, 7]
        for i in range(len(test_n)):
            z = q_model.observe(x=np.expand_dims(x_test[test_n[i]], 0))['z']
            latent = sess.run(z, feed_dict={
                x_input: np.expand_dims(x_test[test_n[i]], 0),
                n: 1,
                n_particles: 100,
                std_noise: 0.7
            })
            x_gen = tf.reshape(gen_model.observe(z=latent)["x_mean"], [-1, 28, 28, 1])
            images = sess.run(x_gen, feed_dict={})
            name = os.path.join(args.result_path, "{}.png".format(i))
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
    
    # Paths
    parser.add_argument('--data_dir', default=os.path.join('data'))
    parser.add_argument('--result_path', default=os.path.join('results'))
    parser.add_argument('--checkpoints_path', default=os.path.join('checkpoints'))
    
    args = parser.parse_args()

    # Select which GPU to use and enable mixed precision
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    train_vae(args)