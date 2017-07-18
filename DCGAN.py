import sys
import pickle
import numpy as np
import tensorflow as tf
import _pickle as cPickle
import os


#weight_variable_funtion1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# bias_function1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convd2d_function1
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# bais_function2
def weight_variable2(shape):
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)
    
#weight_variable_funtion2
def weight_variable3(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

# convd2d_function2
def conv22d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

#image_data(x)
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = np.load(fp, encoding='latin1')
    fp.close()

    return data

X_train = None
y_train = []

X_train=unpickle('pickle-file')
X_image=X_train[0:2000]/ 255 #X_train_shape(1000, 128, 128, 3)

#Generator

inputs = tf.random_uniform([self.batchsize,3], minval=-1.0, maxval=1.0)

def g_inference(inputs):
    depths = [512, 256, 128, 64, 3]
    i_depth = depths[0:4]
    o_depth = depths[1:5]
    with tf.variable_scope('g'):
        w0 = weight_variable([3, i_depth[0] * 4 * 4])
        b0 = bias_variable([i_depth[0]])
        dc0 = tf.nn.bias_add(tf.reshape(tf.matmul(inputs, w0), [-1, 4, 4, i_depth[0]]), b0)
        mean0, variance0 = tf.nn.moments(dc0, [0, 1, 2])
        bn0 = tf.nn.batch_normalization(dc0, mean0, variance0, None, None, 1e-5)
        out = tf.nn.relu(bn0)  #shape=(100, 4, 4, 512)
        # deconvolution layers
        for i in range(4):
            w = weight_variable([5, 5, o_depth[i], i_depth[i]])
            b = bias_variable([o_depth[i]])
            dc = tf.nn.conv2d_transpose(out, w, [2000, 4 * 2 ** (i + 1), 4 * 2 ** (i + 1), o_depth[i]], [1, 2, 2, 1])
            out = tf.nn.bias_add(dc, b)
            if i < 3:
                mean, variance = tf.nn.moments(out, [0, 1, 2])
                out = tf.nn.relu(tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5))
                tf.nn.tanh(out)
    return tf.nn.tanh(out)
          #shape=(2000, 64, 64, 3)



#discriminator

def discriminator(x):
    depths = [3, 64, 128, 256, 512]
    i_depth = depths[0:4]
    o_depth = depths[1:5]
    with tf.variable_scope('d'):
        # convolution layer
        for i in range(4):
            w = weight_variable2([5, 5, i_depth[i], o_depth[i]])
            b = bias_variable([o_depth[i]])
            outputs=tf.reshape(x, [-1,o_depth[i],o_depth[i],i_depth[i]])
            c = tf.nn.bias_add(conv22d(outputs, w), b)
            mean, variance = tf.nn.moments(c, [0, 1, 2])
            bn = tf.nn.batch_normalization(c, mean, variance, None, None, 1e-5)
            outputs = tf.maximum(0.2 * bn, bn)
            # reshepe and fully connect to 2 classes
            dim = 1
            for d in outputs.get_shape()[1:].as_list():
                dim *= d
                w = weight_variable3([dim, 2])
                b = bias_variable([2])
                tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b)
                return tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b)
               #(?, 2)
#Placeholder
output_from_noise = tf.placeholder(tf.float32, [2000, 3])
output_from_given_data = tf.placeholder(tf.float32, [2000, 128, 128, 3])


g_output=g_inference(output_from_noise)

#D(G(z)) z=inputs
logits_from_g = discriminator(g_output)
#D(x) x=image(X_image)
logits_from_i = discriminator(output_from_given_data)

#loss_function
def loss(logits_from_g, logits_from_i):
    d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g,
                                                                          labels=tf.zeros([4096000], dtype=tf.int64)))
    d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_i,
                                                                          labels=tf.ones([16384000], dtype=tf.int64)))
    return [d_loss_fake, d_loss_real]

#optimizer_function
def training(loss, n):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss, var_list=n)
    return train_step
each_variables_parametor
g_vars = [v for v in tf.trainable_variables() if v.name.startswith('g')]
d_vars = [v for v in tf.trainable_variables() if v.name.startswith('d')]


# loss_definition
d_loss_fake, d_loss_real = loss(logits_from_g, logits_from_i)
g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g, 
                                                                      labels=tf.ones([4096000], dtype=tf.int64)))
# training_ops_definition
d_train_op = training(d_loss_fake + d_loss_real,d_vars)
g_train_op = training(g_loss, g_vars)


#convert tensor-inputs(z) to matrix
with tf.Session() as sess:
    noise = inputs.eval()
   

   
sess=tf.Session() 
sess.run(tf.global_variables_initializer())
num_steps = 10
for step in range(num_steps):
    sess.run(g_output, feed_dict = {output_from_noise: noise})
    sess.run(d_train_op, feed_dict = {
    output_from_noise: noise,
    output_from_given_data: X_image})
    sess.run(g_train_op, feed_dict = {output_from_noise: noise})
    if step % 10 == 0:
        d_loss_noise, d_loss_image = sess.run([d_loss_fake,d_loss_real], feed_dict = {
        output_from_noise: noise,
        output_from_given_data: X_image})
        print('step: %d, loss_noise: %f, loss_image: %f'%(step, d_loss_noise, d_loss_image))
