import sys
import pickle
import numpy as np
import tensorflow as tf

# reading image from pickle file
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = np.load(fp, encoding='latin1')
    fp.close()
    return data

X_train=unpickle('seen_batch.pickle')
X_image=X_train/ 255 # shape=(5104, 128, 128, 3)


# Generator

inputs = tf.random_uniform([5104, 100], minval=-1.0, maxval=1.0)

def g_inference(x):
    depths = [250, 150, 90, 54, 3]
    i_depth = depths[0:4]
    o_depth = depths[1:5]
    with tf.variable_scope('g'):
        # reshape from inputs
        with tf.variable_scope('reshape'):
            w0 = tf.get_variable('weights',[100, i_depth[0] * 8 * 8], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            b0 = tf.get_variable('biases', [i_depth[0]], tf.float32, tf.zeros_initializer)
            dc0 = tf.nn.bias_add(tf.reshape(tf.matmul(x, w0), [-1, 8, 8, i_depth[0]]), b0)
            mean0, variance0 = tf.nn.moments(dc0, [0, 1, 2])
            bn0 = tf.nn.batch_normalization(dc0, mean0, variance0, None, None, 1e-5)
            out = tf.nn.relu(bn0)  # shape=(5104, 8, 8, 512)
        # deconvolution layers
        for i in range(4):
            with tf.variable_scope('conv%d' % (i + 1)):
                w = tf.get_variable('weights', [5, 5, o_depth[i], i_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('biases',[o_depth[i]], tf.float32, tf.zeros_initializer)
                dc = tf.nn.conv2d_transpose(out, w, [5104, 8 * 2 ** (i+1), 8 * 2 ** (i+1), o_depth[i]], [1, 2, 2, 1])
                out = tf.nn.bias_add(dc, b)
                if i < 3:
                    mean, variance = tf.nn.moments(out, [0, 1, 2])
                    out = tf.nn.relu(tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5))
                    tf.nn.tanh(out)
    return tf.nn.tanh(out)
          # shape=(5104, 128, 128, 3)


# Discriminator
def discriminator(x):
    depths = [3,128, 256, 512,1024]
    i_depth = depths[0:4]
    o_depth = depths[1:5]
    with tf.variable_scope('d'):
        outputs = x
        # convolution layer
        for i in range(4):
            with tf.variable_scope('conv%d' % i):
                w = tf.get_variable('weights', [5, 5, i_depth[i], o_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('biases', [o_depth[i]], tf.float32, tf.zeros_initializer)
                c = tf.nn.bias_add(tf.nn.conv2d(outputs, w, [1, 2, 2, 1], padding='SAME'), b)
                mean, variance = tf.nn.moments(c, [0, 1, 2])
                bn = tf.nn.batch_normalization(c, mean, variance, None, None, 1e-5)
                outputs = tf.maximum(0.2 * bn, bn)
                # reshepe and fully connect to 2 classes
        with tf.variable_scope('classify'):
            dim =1
            for d in outputs.get_shape()[1:].as_list():
                dim *= d
            w =tf.get_variable('weights',[dim, 2], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable('biases', [2], tf.float32, tf.zeros_initializer)
            tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b)
    return tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b)

# TO DO: not difined discriminator(x) and g_inference(x) before definition, d_vars and g_vars is empty
# each variable in Discriminator
d_vars = [v for v in tf.trainable_variables() if v.name.startswith('d')]

# each variable in Generator
g_vars = [v for v in tf.trainable_variables() if v.name.startswith('g')]

# '''image and input(z) placeholder'''
z = tf.placeholder(tf.float32, [5104, 100])
image = tf.placeholder(tf.float32, [5104, 128, 128, 3])



# '''G(z), D(G(z)), D(x)'''
# G(z)
g_output=g_inference(z)

tf.get_variable_scope().reuse_variables() # reuse tf.get_variable
# D(G(z))
logits_from_g = discriminator(g_output)

# D(x)
logits_from_i = discriminator(image)




# loss fuction
def loss(logits_from_g, logits_from_i):
    loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g,
                                                                          labels=tf.zeros([5104], dtype=tf.int64)))
    loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_i,
                                                                          labels=tf.ones([5104], dtype=tf.int64)))
    return [loss_1, loss_2]



# training function
def training(_loss, n):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(_loss, var_list=n)
    return train_step



# loss difinition
d_loss_fake, d_loss_real = loss(logits_from_g, logits_from_i)
g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g, 
                                                                      labels=tf.ones([5104], dtype=tf.int64)))


# training op
d_train_op = training(d_loss_fake + d_loss_real,d_vars)
g_train_op = training(g_loss,g_vars)


# TO DO: sess is being redifinde, causes an error
# convert input from tensor to matrix
with tf.Session() as sess:
    noise = inputs.eval()  # noise.shape=(5104, 100)
    

    
# activation of training
sess=tf.Session()
sess.run(tf.global_variables_initializer())
num_steps = 1
for step in range(num_steps):
    sess.run(d_train_op, feed_dict = {z: noise, image: X_image})
    sess.run(g_train_op, feed_dict = {z: noise})
    if step % 1 == 0:
        d_loss_noise, d_loss_image = sess.run([d_loss_fake,d_loss_real], feed_dict = {
        z: noise,
        image: X_image})
        print('step: %d, loss_noise: %f, loss_image: %f'%(step, d_loss_noise, d_loss_image))
