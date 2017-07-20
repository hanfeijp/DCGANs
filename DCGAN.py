import sys
import pickle
import numpy as np
import tensorflow as tf
import _pickle as cPickle
import os



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス変数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#画像読み込み
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = np.load(fp, encoding='latin1')
    fp.close()
    return data


X_train=unpickle('seen_batch.pickle')


X_image=X_train/ 255 #X_train_shape(400, 64, 64, 3)

#Generator

inputs = tf.random_uniform([5072, 3], minval=-1.0, maxval=1.0)

def g_inference(inputs):
    depths = depths = [250, 150, 90, 54, 3]
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
            dc = tf.nn.conv2d_transpose(out, w, [5072, 4 * 2 ** (i + 1), 4 * 2 ** (i + 1), o_depth[i]], [1, 2, 2, 1])
            out = tf.nn.bias_add(dc, b)
            if i < 3:
                mean, variance = tf.nn.moments(out, [0, 1, 2])
                out = tf.nn.relu(tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5))
                tf.nn.tanh(out)
    return tf.nn.tanh(out)
          #shape=shape=(400, 64, 64, 3)




#discriminator

def activation_summary(x):
    '''
    Add histogram and sparsity summaries of a tensor to tensorboard
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def discriminator(x):
    # 畳み込み１層目(conv1)
    with tf.variable_scope('d'):
        W_conv1 = weight_variable([5, 5, 3, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1) #(10000, 16, 16, 32)
        tf.summary.image('conv1', x, 10)
        norm1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
        outputs = tf.maximum(0.2 * norm1, norm1)
        
        W_conv2 = weight_variable([5, 5, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
        activation_summary(h_conv2)
        # norm2 : 局所反応正規化 (local response normalization)
        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
        outputs1 = tf.maximum(0.2 * norm2, norm2)
        dim = 1
        for d in outputs.get_shape()[1:].as_list():
            dim *= d
            w = weight_variable([dim, 2])
            b = bias_variable([2])
            tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b)
    return tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b)



#Placeholder
output_from_noise = tf.placeholder(tf.float32, [None, 3])
output_from_given_data = tf.placeholder(tf.float32, [None, 64, 64, 3])

g_output=g_inference(output_from_noise)
#D(G(z)) z=inputs
logits_from_g = discriminator(g_output)
#D(x) x=画像(X_image)
logits_from_i = discriminator(output_from_given_data)

#損失関数
def loss(logits_from_g, logits_from_i):
    d_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g,
                                                                          labels=tf.zeros([5072], dtype=tf.int64)))
    d_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_i,
                                                                          labels=tf.ones([5072], dtype=tf.int64)))
    return [d_loss_fake, d_loss_real]

#勾配
def training(loss, n):
    train_step = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.5).minimize(loss, var_list=n)
    return train_step

g_vars = [v for v in tf.trainable_variables() if v.name.startswith('g')]
d_vars = [v for v in tf.trainable_variables() if v.name.startswith('d')]


# loss
d_loss_fake, d_loss_real = loss(logits_from_g, logits_from_i)
g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g, 
                                                                      labels=tf.ones([5072], dtype=tf.int64)))
# training
d_train_op = training(d_loss_fake + d_loss_real,d_vars)
g_train_op = training(g_loss, g_vars)

with tf.Session() as sess:
    noise = inputs.eval()
    
sess=tf.Session()
sess.run(tf.global_variables_initializer())
num_steps = 200
for step in range(num_steps):
    sess.run(g_output, feed_dict = {output_from_noise: noise})
    sess.run(d_train_op, feed_dict = {
    output_from_noise: noise,
    output_from_given_data: X_image})
    sess.run(g_train_op, feed_dict = {output_from_noise: noise})
    if step % 1 == 0:
        d_loss_noise, d_loss_image = sess.run([d_loss_fake,d_loss_real], feed_dict = {
        output_from_noise: noise,
        output_from_given_data: X_image})
        print('step: %d, loss_noise: %f, loss_image: %f'%(step, d_loss_noise, d_loss_image))
