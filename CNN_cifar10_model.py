from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf
import os

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


# In[10]:
def distorted_inputs(data_dir):
    
    filenames = [os.path.join(data_dir, 'data_batch_1')]  #['train_batch.pickle']
    filename_queue = tf.train.string_input_producer(filenames)  #<tensorflow.python.ops.data_flow_ops.FIFOQueue at 0x112bf93c8>
    
    read_input = read_cifar10(filename_queue) # <__main__.read_cifar10.<locals>.CIFAR10Record at 0x112df2ef0>
    reshaped_image = tf.cast(read_input.uint8image, tf.float32) # <tf.Tensor 'Cast_9:0' shape=(128, 128, 3) dtype=float32>
    
    height = 24
    width = 24
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3]) # <tf.Tensor 'random_crop:0' shape=(96, 96, 3) dtype=float32>
    distorted_image = tf.image.random_flip_left_right(distorted_image) # <tf.Tensor 'cond/Merge:0' shape=(96, 96, 3) dtype=float32>
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63) # 'adjust_brightness/Identity_1:0' shape=(96, 96, 3)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8) # 'adjust_contrast/Identity_1:0' shape=(96, 96, 3)
    
    float_image = tf.image.per_image_standardization(distorted_image) # <tf.Tensor 'div:0' shape=(96, 96, 3) dtype=float32>
    float_image.set_shape([height, width, 3]) # 'div_2:0' shape=(96, 96, 3)
    read_input.label.set_shape([1]) # <__main__.read_cifar10.<locals>.CIFAR10Record at 0x1136ab278>
    min_queue_examples = int(10000 *0.4) # 4000

    # Filling queue with 12000 CIFAR images before starting to train. This will take a few minutes.
    # shuffle_batch:0' shape=(128, 96, 96, 3) dtype=float32>, shape=(128,) dtype=int32>)
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, 128, shuffle=True)  
       


# In[11]:


def read_cifar10(filename_queue):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth # 49152
  
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes) # #<tensorflow.python.ops.io_ops.FixedLengthRecordReader at 0x112c44128>
    result.key, value = reader.read(filename_queue) # # key:<tf.Tensor 'ReaderReadV2:0' shape=() dtype=string> vaule:ReaderReadV2:1'

    record_bytes = tf.decode_raw(value, tf.uint8) # <tf.Tensor 'DecodeRaw:0' shape=(?,) dtype=uint8>
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32) # # <tf.Tensor 'Cast:0' shape=(?,) dtype=int32>

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),[result.depth, result.height, result.width]) # # <tf.Tensor 'Reshape:0' shape=(3, 128, 128) dtype=uint8>
    result.uint8image = tf.transpose(depth_major, [1, 2, 0]) # <tf.Tensor 'transpose_1:0' shape=(128, 128, 3) dtype=uint8>

    return result


# In[21]:


def inputs(data_dir):
    """Construct input for evaluation using the Reader ops."""
    filenames = [os.path.join(data_dir, 'test_batch')]
    num_examples_per_epoch = 10000
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = 24
    width = 24
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_queue_examples = int(num_examples_per_epoch*0.4)

   # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, 128, shuffle=False)


# In[13]:


def cnn(x):
    BATCH_SIZE = 128
    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(x):
        tensor_name = x.op.name
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name='conv1')
        _activation_summary(conv1)
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name='conv2')
        _activation_summary(conv2)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[128], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name='conv3')
        _activation_summary(conv3)
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('conv4') as scope:
        kernel = tf.get_variable('weights', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name='conv4')
        _activation_summary(conv4)
    norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
    pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    with tf.variable_scope('fc5') as scope:
        dim = 1
        for d in pool4.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool4, [BATCH_SIZE, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 1024], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        fc5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weights), biases), name='fc5')
        _activation_summary(fc5)

    with tf.variable_scope('fc6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 256], stddev=0.02, wd=0.005)
        biases = tf.get_variable('biases', shape=[256], initializer=tf.constant_initializer(0.0))
        fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc5, weights), biases), name='fc6')
        _activation_summary(fc6)

    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable('weights', shape=[256, 10], initializer=tf.truncated_normal_initializer(stddev=0.02))
        biases = tf.get_variable('biases', shape=[10], initializer=tf.constant_initializer(0.0))
        fc7 = tf.nn.bias_add(tf.matmul(fc6, weights), biases, name='fc7')
        _activation_summary(fc7)

    return fc7   #shape=(BATCH_SIZE, 3)


# In[14]:


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy'))
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight(L2 loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



# In[15]:


def train(total_loss, global_step):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')

    loss_averages_op = loss_averages.apply(losses + [total_loss]) #op for generating moving averages of losses.
    # Variables that affect learning rate.
    num_batches_per_epoch = 10000 / 128
    decay_steps = int(num_batches_per_epoch * 10000)
    lr = tf.train.exponential_decay(0.1, global_step, decay_steps, 0.1, staircase=True)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr) # instead of tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op #op for training.


# In[22]:


with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    images, labels = inputs('/Users/hagiharatatsuya/Downloads/cifar-10-batches-py/')
    c_logits = cnn(images)
    loss = loss(c_logits, labels)
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
                sec_per_batch = float(duration / FLAGS.log_frequency)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(hooks=[tf.train.StopAtStepHook(last_step=1000),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=False)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)
