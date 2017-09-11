import sys
import pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.misc
import time

import tensorflow.contrib.slim as slim


z = tf.placeholder(tf.float32, [None, 100])
image = tf.placeholder(tf.float32, [64, 64, 64, 3])


sample_z = np.random.uniform(-1, 1, size=(64, 100))
batch_z = np.random.uniform(-1, 1, [64, 100])



def show_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
    
    
def batch_norm(x, name, train=True):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x,
                      decay=0.9, 
                      updates_collections=None,
                      epsilon=1e-5,
                      scale=True,
                      is_training=train,
                      scope=name)
    
    

def conv2d(input_, output_dim, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [5, 5, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, 2, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


        
def lrelu(x, name="lrelu"):
    return tf.maximum(x, 0.2*x)



def linear(input_, output_size,scope=None, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias




def deconv2d(input_, output_shape, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [5, 5, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, 2, 2, 1])
        b = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        if with_w:
            return deconv, w, b
        else:
            return deconv

       


# In[2]: # read image form pickle file


def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


test= unpickle("train_image.pickle")


X_image=np.array(test)/255
X_image.shape


# In[3]: # discriminator, generator and sampler
# TO DO : to reduce amount of calculation, you change generator as well as sampler like this
#z, h0_w, h0_b = linear(z_, g_fc*8*8*4, 'g_h0_lin',with_w=True)
            #h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 8, 8, g_fc*4]), 'g_bn0'))
            #h1, h1_w, h1_b = deconv2d(h0, [batch_size, 16, 16, g_fc*3], name='g_h1', with_w=True
            #h2, h2_w, h2_b = deconv2d(h1, [batch_size, 32, 32, g_fc*2], name='g_h2', with_w=True)
            #h3, h3_w, h3_b = deconv2d(h2, [batch_size, 64, 64, g_fc*1], name='g_h3', with_w=True)
         

def discriminator():
    reuse = False
    def model(image):
        nonlocal reuse
        batch_size=64
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            h0 = lrelu(conv2d(image, 64, name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(h0, 128, name='d_h1_conv'),'d_bn1'))
            h2 = lrelu(batch_norm(conv2d(h1, 256, name='d_h2_conv'),'d_bn2'))
            h3 = lrelu(batch_norm(conv2d(h2, 512, name='d_h3_conv'),'d_bn3'))  # shape=(batch_size, 64, 64, 3)　
            h4 = linear(tf.reshape(h3, [batch_size, -1]),1,'d_h4_lin')
        reuse = True
        return tf.nn.sigmoid(h4), h4
    return model

    

                    
    # shape=(batch_size, 64, 64, 3)
def generator():
    reuse = False
    def model(z_):
        nonlocal reuse
        batch_size=64
        with tf.variable_scope("generator", reuse=reuse) as scope:
            # project `z` and reshape
            z, h0_w, h0_b = linear(z_, 64*8*4*4, 'g_h0_lin',with_w=True)
            h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 4, 4, 64*8]), 'g_bn0'))
            h1, h1_w, h1_b = deconv2d(h0, [batch_size, 8, 8, 64*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(batch_norm(h1, 'g_bn1'))
            h2, h2_w, h2_b = deconv2d(h1, [batch_size, 16, 16, 64*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(batch_norm(h2, 'g_bn2'))
            h3, h3_w, h3_b = deconv2d(h2, [batch_size, 32, 32, 64*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(batch_norm(h3, 'g_bn3'))
            h4, h4_w, h4_b = deconv2d(h3, [batch_size, 64, 64, 3], name='g_h4', with_w=True)
        reuse = True
        return tf.nn.tanh(h4)  #shape=(batch_size, 64, 64, 3)
    return model




def sampler():# shape=(batch_size, 64, 64, 3)　
    reuse = True
    def model(z_):
        nonlocal reuse
        batch_size=64
        with tf.variable_scope("generator",reuse=reuse) as scope:
            # project `z` and reshape
            z= linear(z_, 64*8*4*4,'g_h0_lin')
            h0 = tf.nn.relu(batch_norm(tf.reshape(z, [-1, 4, 4, 64*8]),'g_bn0',train=False))
            h1 = deconv2d(h0, [batch_size, 8, 8, 64*4], name='g_h1')
            h1 = tf.nn.relu(batch_norm(h1,'g_bn1',train=False))
            h2 = deconv2d(h1, [batch_size, 16, 16, 64*2], name='g_h2')
            h2 = tf.nn.relu(batch_norm(h2,'g_bn2',train=False))
            h3 = deconv2d(h2, [batch_size, 32, 32, 64*1], name='g_h3')
            h3 = tf.nn.relu(batch_norm(h3,'g_bn3',train=False))
            h4 = deconv2d(h3, [batch_size, 64, 64, 3], name='g_h4')
        return tf.nn.tanh(h4)  #shape=(batch_size, 64, 64, 3)
    return model




g = generator()
d = discriminator()
s = sampler()


G=g(z)  #G(z)
D, D_logits = d(image) #D(x)

sampler = s(z)
D_, D_logits_ = d(G)   #D(G(z))


# In[4]: # loss function




d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

d_loss = d_loss_real + d_loss_fake


# In[5]:  # optim


d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
                             
                             
g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)



# In[ ]: # train

saver=tf.train.Saver()
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
    sess.run(tf.global_variables_initializer())
    sample_files=X_image[0:64]
    sample = [sample_file for sample_file in sample_files]
    sample_images = np.array(sample).astype(np.float32)
    
    counter=1
    epochs=100
    start_time=time.time()
    show_variables()
    
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
            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time:%4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,
                                                                                         time.time()-start_time, errD_fake+errD_real, errG))
            
            # show sample image while trainig
            if np.mod(counter, 30)==1:
                samples, d_loss_sample, g_loss_sample = sess.run([sampler, d_loss, g_loss],
                                               feed_dict={z: sample_z, image: sample_images})
                
                print("[Sample] d_loss:%.8f, g_loss:%.8f" % (d_loss_sample, g_loss_sample))
                col=8
                rows=[]
                for i in range(8):
                    rows.append(np.hstack(samples[col * i + 0:col * i + col]))
                vnari=np.vstack(rows)
                plt.imshow(vnari)
                plt.show()
            # save sess to directory
            if np.mod(counter, 2)==1:
                saver.save(sess, "/Users/hagiharatatsuya/Downloads/dcgan_dir/decgan")
                             
                             
                  
# In[ ]: # Restoring　session from directory and move again                            
                                               
def move_once(saver):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('/Users/hagiharatatsuya/Downloads/dcgan_dir')
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        
        sample_files=X_image[0:64]
        sample = [sample_file for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
    
        counter=1
        epochs=1
        start_time=time.time()
        show_variables()
    
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
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time:%4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs,
                                                                                         time.time()-start_time, errD_fake+errD_real, errG))
                if np.mod(counter, 10)==1:
                    samples, d_loss_sample, g_loss_sample = sess.run([sampler, d_loss, g_loss],
                                               feed_dict={z: sample_z, image: sample_images})
                
                    print("[Sample] d_loss:%.8f, g_loss:%.8f" % (d_loss_sample, g_loss_sample))
                    col=8
                    rows=[]
                    for i in range(8):
                        rows.append(np.hstack(samples[col * i + 0:col * i + col]))
                    vnari=np.vstack(rows)
                    plt.imshow(vnari)
                    plt.show()
                if np.mod(counter, 10)==1:
                    saver.save(sess, '/Users/hagiharatatsuya/Downloads/dcgan_dir/ckpt')

                             
# In[ ]:  
                             
saver = tf.train.Saver()
move_once(saver)
