from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import os
import math
import numpy as np


def inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir)]
    num_examples_per_epoch = 540

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
    imgin = tf.reshape(tf.decode_raw(features["image"], tf.uint8), tf.stack([96, 96, 3]))
    
    reshaped_image = tf.cast(imgin, tf.float32)
    height = 64
    width = 64
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    float_image = tf.image.per_image_standardization(resized_image)
    float_image.set_shape([height, width, 3])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)
    
    num_preprocess_threads = 16
    images, label_batch = tf.train.batch([float_image, label], batch_size=batch_size,
        num_threads=num_preprocess_threads,capacity=min_queue_examples + 3 * batch_size)
    
    tf.summary.image('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# In[2]:

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
        tf.summary.histogram(tensor_name+'/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 32], stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name='conv1')
        _activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 32, 64],stddev=0.1,wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name='conv2')
        _activation_summary(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 64, 128],stddev=0.1,wd=0.0)
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name='conv3')
        _activation_summary(conv3)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[3, 3, 128, 256],stddev=5e-2,wd=0.0)
        conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name='conv4')
        _activation_summary(conv4)
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    norm4 = tf.nn.lrn(pool4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
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
        weights = _variable_with_weight_decay('weights', [256, 4], stddev=0.02, wd=0.0)
        biases = tf.get_variable('biases', shape=[4], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name='fc7')
        _activation_summary(fc7)

    return fc7   # shape=(BATCH_SIZE, 4)



# In[3]: # for evaluation


def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('/Users/Downloads/cnn_tbdir')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(540 / 128))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * 128
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
                
                # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
 
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
            
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)



# In[ ]: # Session


with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, labels = inputs('/Users/Downloads/CNN, DCGANコード/cnn_test.tfrecords',128)
    
    logits = cnn(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('/Users/hagiharatatsuya/Downloads/eval_dir', g)
    while True:
        eval_once(saver,summary_writer, top_k_op, summary_op)
        if False:
            break
        time.sleep(60*5)# how often eval
