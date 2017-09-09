from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import os
import math
import numpy as np



def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
      features={"label": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string)})
    label = tf.cast(features["label"], tf.int32)
    imgin = tf.reshape(tf.decode_raw(features["image"], tf.uint8), tf.stack([128, 128, 3]))
    reshaped_image = tf.cast(imgin, tf.float32)

    height = 96
    width = 96

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)
    float_image.set_shape([height, width, 3])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(30000*min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
    
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([float_image, label], batch_size=batch_size,
        num_threads=num_preprocess_threads,capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    
    return images, tf.reshape(label_batch, [batch_size])


# In[2]: # CNN like VGGnet which is NN for classifying images


def cnn(x):
    BATCH_SIZE = 128
    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(x):
        tensor_name = x.op.name
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 32], stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name='conv1')
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 32, 64],stddev=0.1,wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name='conv2')
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 64, 128],stddev=0.1,wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name='conv3')
        _activation_summary(conv3)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 128, 256],stddev=5e-2,wd=0.0)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name='conv4')
        _activation_summary(conv4)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5') as scope:
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [BATCH_SIZE, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024],stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name='fc5')
        _activation_summary(fc5)

    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256],stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name='fc6')
        _activation_summary(fc6)

    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', [256, 3], stddev=0.02, wd=0.0)
        biases = tf.get_variable('biases', shape=[3], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name='fc7')
        _activation_summary(fc7)

    return fc7   #shape=(BATCH_SIZE, 3)


# In[3]: # loss


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight(L2 loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# In[4]: # train using AdamOptimizer


def train(total_loss, global_step):
    
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss]) 
    
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer() # instead of tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op #op for training.


# In[ ]: # Session


with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    # for train
    train_image, train_label = distorted_inputs('/Users/hagiharatatsuya/Downloads/manshion_train.tfrecords',128)
    c_logits = cnn(train_image)
    loss = loss(c_logits, train_label)
    train_op = train(loss, global_step)
    
    class _LoggerHook(tf.train.SessionRunHook):
        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.

        def after_run(self, run_context, run_values):
            if self._step % 10 == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = 10 * 128 / duration
                sec_per_batch = float(duration / 10)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    # checkpoint_dir must be directory
    with tf.train.MonitoredTrainingSession(checkpoint_dir='/Users/hagiharatatsuya/Downloads/check_point',
                                           hooks=[tf.train.StopAtStepHook(last_step=15000),
               tf.train.NanTensorHook(loss), _LoggerHook()], config=tf.ConfigProto(log_device_placement=False)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)
