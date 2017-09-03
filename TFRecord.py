from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import sys
import pickle
import numpy as np

# read data to input to TFRecord
def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data



test_images=unpickle('test_batch')['data']
test_labels=unpickle('test_batch')['labels']

# In[2]:


# weite to TFRecordfile(the file's extension is 'tfrecords')

writer = tf.python_io.TFRecordWriter('/Users/hagiharatatsuya/Downloads/test_batch.tfrecords')
for i in range(0, len(test_images)):
    record = tf.train.Example(features=tf.train.Features(feature={
          "label": tf.train.Feature(
              int64_list=tf.train.Int64List(value=[test_labels[i]])),
          "image": tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[np.array(test_images[i]).tostring()]))
    }))
    
writer.write(record.SerializeToString())
writer.close()


# In[3]: # confirm the file's contents


file_name_queue = tf.train.string_input_producer(['/Users/hagiharatatsuya/Downloads/test_batch.tfrecords'])

reader = tf.TFRecordReader()
 
# serialize
_, serialized_examples = reader.read(file_name_queue)
feature = tf.parse_single_example(serialized_examples, features={"label": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string)})



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        imgin = tf.reshape(tf.decode_raw(feature["image"], tf.uint8),
                           tf.stack([32, 32, 3])).eval()
    finally:
        coord.request_stop()
        coord.join(threads)
imgin
