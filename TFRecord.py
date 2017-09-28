from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import _pickle as cPickle


# In[2]: for pickle file


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

# for one-hot

def dense_to_one_hot(labels_dense, num_classes):
    for_cnn3=np.array(labels_dense)
    num_classes=10
    num_labels = for_cnn3.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + for_cnn3.ravel()-1] = 1
    return labels_one_hot



# In[3]: # train_image and train_label are cifar-10 data


train_image=np.concatenate((im1,im2,im3,im4,im5))
tr_im=tuple(train_image)

train_label=la1+la2+la3+la4+la5
tr_la=tuple(train_label)


# In[4]:

listed=[]
for i, v in zip(tr_im, tr_la):
    listed.append([v,i])
    

# In[5]: writing data to TFRecords file


writer = tf.python_io.TFRecordWriter('/Users/hagiharatatsuya/Downloads/sample.tfrecords')
for label, img in listed:
    record = tf.train.Example(features=tf.train.Features(feature={
          "label": tf.train.Feature(
              int64_list=tf.train.Int64List(value=[label])),
          "image": tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[img.tostring()]))
      }))
 
    writer.write(record.SerializeToString())
writer.close()


# In[6]:


file_name_queue = tf.train.string_input_producer(['/Users/hagiharatatsuya/Downloads/sample.tfrecords'])

reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_name_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          "label": tf.FixedLenFeature([], tf.int64),
          "image": tf.FixedLenFeature([], tf.string)
      })
label = tf.cast(features["label"], tf.int32)
imgin = tf.reshape(tf.decode_raw(features["image"], tf.uint8),
                           tf.stack([32, 32, 3]))



# In[7]: making 128 size mini batch


images, labels = tf.train.batch(
      [imgin, label], batch_size=128, num_threads=2,
      capacity=1000 + 3 * 128)


# In[8]: confirming data contents

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        print(sess.run(images))
    finally:
        coord.request_stop()
    coord.join(threads)
