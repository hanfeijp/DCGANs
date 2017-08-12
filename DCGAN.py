# coding: utf-8

# In[1]:  # build model

import sys
import pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.misc
import time


z = tf.placeholder(tf.float32, [None, 100])
image = tf.placeholder(tf.float32, [64, 64, 64, 3])


sample_z = np.random.uniform(-1, 1, size=(100, 100))
batch_z = np.random.uniform(-1, 1, [64, 100])


def batch_norm(c):
    mean, variance = tf.nn.moments(c, [0, 1, 2])
    bn = tf.nn.batch_normalization(c, mean, variance, None, None, 1e-5)
    return bn

def conv2d(input_, output_dim, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [5, 5, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    
def lrelu(x, name="lrelu"):
    return tf.maximum(x, 0.2*x)

def linear(input_, output_size):
    shape = input_.get_shape().as_list()
    with tf.variable_scope("Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(input_, matrix),bias)

def deconv2d(input_, output_shape, name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [5, 5, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, 2, 2, 1])
        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())




# In[2]:  # read image form pickle file

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

img_batch= unpickle("62_seen_batch.pickle")
X_image=np.array(img_batch)/ 255
# shape=(batch_size, 64, 64, 3)


# In[8]:    discriminator, generator and sampler

def discriminator(image):
    batch_size=64
    with tf.variable_scope("discriminator") as scope:
        h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(h0, 128, name='d_h1_conv')))
        h2 = lrelu(batch_norm(conv2d(h1, 256, name='d_h2_conv')))
        h3 = lrelu(batch_norm(conv2d(h2, 512, name='d_h3_conv')))
        return linear(tf.reshape(h3, [batch_size, -1]), 2) # shape=(batch_size, 64, 64, 3)　
    
    
    
def generator(z_):# shape=(batch_size, 64, 64, 3)
    batch_size=64
    with tf.variable_scope("generator") as scope:
        # project `z` and reshape
        z= linear(z_, 64*8*4*4)
        h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 4, 4, 64*8])))
        h1 = tf.nn.relu(batch_norm(deconv2d(h0, [batch_size, 8, 8, 64*4], name='g_h1')))
        h2 = tf.nn.relu(batch_norm(deconv2d(h1, [batch_size, 16, 16, 64*2], name='g_h2')))
        h3 = tf.nn.relu(batch_norm(deconv2d(h2, [batch_size, 32, 32, 64*1], name='g_h3')))
        h4 = deconv2d(h3, [batch_size, 64, 64, 3], name='g_h4')
    return tf.nn.tanh(h4)  #shape=(batch_size, 64, 64, 3)



def sampler(z_):# shape=(batch_size, 64, 64, 3)　
    batch_size=100
    with tf.variable_scope("sampler") as scope:
        # project `z` and reshape
        z= linear(z_, 64*8*4*4)
        h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 4, 4, 64*8])))
        h1 = tf.nn.relu(batch_norm(deconv2d(h0, [batch_size, 8, 8, 64*4], name='g_h1')))
        h2 = tf.nn.relu(batch_norm(deconv2d(h1, [batch_size, 16, 16, 64*2], name='g_h2')))
        h3 = tf.nn.relu(batch_norm(deconv2d(h2, [batch_size, 32, 32, 64*1], name='g_h3')))
        h4 = deconv2d(h3, [batch_size, 64, 64, 3], name='g_h4')
    return tf.nn.tanh(h4)  #shape=(batch_size, 64, 64, 3)



G=generator(z)  #G(z)
D_logits = discriminator(image) #D(x)
sampler = sampler(z)


# In[9]:

tf.get_variable_scope().reuse_variables()
D_logits_ = discriminator(G)   #D(G(z))


# In[10]: # loss function
batch_label=64

d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_logits, labels=tf.ones([batch_label], dtype=tf.int64)))
d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros([batch_label], dtype=tf.int64)))


g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones([batch_label], dtype=tf.int64)))
d_loss = d_loss_real + d_loss_fake


# In[11]:  # optim

d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

d_optim = tf.train.GradientDescentOptimizer(0.0001).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.GradientDescentOptimizer(0.0001).minimize(g_loss, var_list=g_vars)



# In[13]:    # train

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

start_time=time.time()
with tf.Session(config=run_config) as sess:
    sess.run(tf.global_variables_initializer())
    epochs=25
    for epoch in range(epochs):
        batch_idxs= min (len(X_image), np.inf) // 64
        for idx in range (0, batch_idxs):
            bacth_files= X_image[idx*64:(idx+1)*64]
            batch = [batch_file for batch_file in bacth_files]
            batch_images = np.array(batch).astype(np.float32)
            sess.run(d_optim, feed_dict = {z: batch_z, image: batch_images})
            sess.run(g_optim, feed_dict = {z: batch_z})
        
            # Run g_optim twice to realize loss value
            sess.run(g_optim, feed_dict = {z: batch_z})
            errD_fake = d_loss_fake.eval({z: batch_z })
            errD_real = d_loss_real.eval({image: batch_images})
            errG = g_loss.eval({z: batch_z})
            print("Epoch: [%2d] [%4d/%4d] time:%4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,
                                                                                         time.time()-start_time, errD_fake+errD_real, errG))
        
        # to save image to your local folder
        # TODO:at line 181, you specify your local folder. Please see details of how to use 'cv2.imwrite'
        if epoch%5==0:
            samples = sess.run(sampler,feed_dict={z: sample_z})
            col=8
            rows=[]
            for i in range(8):
                rows.append(np.hstack(samples[col * i + 0:col * i + col]))
                vnari=np.vstack(rows)
                scipy.misc.imsave('/Users/hagiharatatsuya/Downloads/sampler.html/sampler%s.png'% (epoch), vnari)
