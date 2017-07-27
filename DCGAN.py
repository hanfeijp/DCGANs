import sys
import pickle
import numpy as np
import tensorflow as tf

# TODO: first copy and paste unitil line63
# Generator

inputs = tf.random_uniform([1000, 100], minval=-1.0, maxval=1.0)

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
            out = tf.nn.relu(bn0)  #shape=(100, 8, 8, 512)
        # deconvolution layers
        for i in range(4):
            with tf.variable_scope('conv%d' % (i + 1)):
                w = tf.get_variable('weights', [5, 5, o_depth[i], i_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('biases',[o_depth[i]], tf.float32, tf.zeros_initializer)
                dc = tf.nn.conv2d_transpose(out, w, [1000, 8 * 2 ** (i+1), 8 * 2 ** (i+1), o_depth[i]], [1, 2, 2, 1])
                out = tf.nn.bias_add(dc, b)
                if i < 3:
                    mean, variance = tf.nn.moments(out, [0, 1, 2])
                    out = tf.nn.relu(tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5))
                    tf.nn.tanh(out)
    return tf.nn.tanh(out)
          #shape=shape=(400, 64, 64, 3)


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


# TODO: 2th line 67 and 68 copy and paste and activate(push shift and enter key)
x= tf.placeholder(tf.float32, [1000,128,128, 3])
discriminator(x)


# TODO: 3th line72 73 acitivate(d_vars has value) 
d_vars = [v for v in tf.trainable_variables() if v.name.startswith('d')]
d_vars


# TODO: 4th iamge setting line76 to 88(this time amount of images are 1000)
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = np.load(fp, encoding='latin1')
    fp.close()
    return data

X_train=unpickle('seen_batch.pickle')
X_image=X_train/ 255
X_image=X_image[:1000]


# TODO:5th copy and paste line91 to 95 and activate line95
z = tf.placeholder(tf.float32, [None, 100])
image = tf.placeholder(tf.float32, [None, 128, 128, 3])
g_output=g_inference(z)
g_output


# TODO: 6th copy and paste line99 to 101
tf.get_variable_scope().reuse_variables()
logits_from_g = discriminator(g_output)
logits_from_i = discriminator(image)

# TODO: 7th line104 to 110 and acitivate 109
def loss(logits_from_g, logits_from_i):
    loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g,labels=tf.zeros([1000], dtype=tf.int64)))
    loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_i,labels=tf.ones([1000], dtype=tf.int64)))
    return [loss_1, loss_2]

g_vars = [v for v in tf.trainable_variables() if v.name.startswith('g')]



# TODO: 8th copy and paste line114 to 121
# 勾配
def training(loss, n):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss, var_list=n)
    return train_step

# loss
d_loss_fake, d_loss_real = loss(logits_from_g, logits_from_i)
g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_from_g,labels=tf.ones([1000], dtype=tf.int64)))
                     

    
# TODO: 9th copy and paste line126 to 128
# training 
d_train_op = training(d_loss_fake + d_loss_real,d_vars)
g_train_op = training(g_loss,g_vars)



# TODO: 10th copy and paste line 133 to 134 and activate 133 and 134
with tf.Session() as sess:
    noise = inputs.eval()
    
    
# TODO: 11th finally copy and paste line138 to 148 and activate
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
