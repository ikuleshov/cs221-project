
'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from PIL import Image
from os.path import isfile, join
import scipy.misc as misc
import numpy
import sys
import math

# Import MNIST data
import input_data
mnist = input_data.read_data_sets()

# Parameters
learning_rate = 0.1
training_iters = 200000
batch_size = 24
display_step = 10

# Network Parameters
input_width = input_data.image_width
label_width = input_data.label_width
n_input = input_width*input_width*3 # MNIST data input (img shape: 28*28)
n_classes = label_width*label_width*3 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, input_width, input_width, 3])
y = tf.placeholder(tf.float32, [batch_size, label_width * label_width * 3])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def CrossEntropyLoss(pred, label):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, label))
    return (cost, cost)

def L2Loss(pred, label):
    cost = tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred, label)))
    accuracy = tf.div(tf.sqrt(tf.mul(tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred, label))), 2.0)), batch_size)
    return (cost, accuracy)

def PsnrLoss(pred, label):
    cost = tf.reduce_mean(tf.square(tf.sub(pred, label)))
    #cost = tf.reduce_mean(tf.sqrt(tf.square(tf.sub(pred, label))))
    cost = tf.div(1.0, cost)
    #tf.Print(cost, [cost], 'Cost is ')
    cost = 8.68588 * tf.log(cost)
    return (cost, cost)

# Create model
def conv_net(x, weights, biases, keep_prob):
    # Deconvolution Layer
    print ('x ' + str(x.get_shape()))
#     pool = maxpool2d(x, k=4)
#     deconv = tf.nn.conv2d_transpose(pool, weights['deconv1'], [batch_size, label_width, label_width, 3], [1, 1, 1, 1], 'SAME')
#     print ('deconv ' + str(deconv.get_shape()))
    conv1 = tf.nn.depthwise_conv2d(x, weights['conv1'], [1, 1, 1, 1], 'SAME')
    out = tf.reshape(conv1, [batch_size, label_width, label_width, 3])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'conv1': tf.Variable(tf.random_normal([5, 5, 3, 1], mean=1.0, stddev=0)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([2, 2, 32, 64], mean=1.0, stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32, 32, 3, 64], mean=1.0, stddev=0.1)),
    'deconv1': tf.Variable(tf.random_normal([label_width, label_width, 3, 3], mean=1.0, stddev=0)),
#     # 1024 inputs, 10 outputs (class prediction)
#     'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([3])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([n_classes])),
    #'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
print ('Creating conv net')
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
print ('Adding the cost and optimizer')
pred2 = tf.reshape(pred, [batch_size, label_width * label_width * 3])
(cost, accuracy) = L2Loss(pred2, y)
print ('Cost shape: ' + str(cost.get_shape()))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
print ('Initializing variables')
init = tf.initialize_all_variables()

def PrintBatch(batch, width, name):
    labp = batch.reshape(batch_size, width, width, 3)
    labp = numpy.split(labp, batch_size, 0)
    for i in xrange(batch_size):
        lab = labp[i]
        lab = lab.reshape(width, width, 3)
        misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/foo_' + name + str(i) + ".jpeg", lab)

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Launch the graph
print ('Launching session')
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #print ('Processing ' + str(step))
        batch_x, batch_y = mnist.train.next_batch(batch_size)
#         sys.exit()
        #print ('Batch x size ' + str(batch_x.shape))
        #print ('Batch y size ' + str(batch_y.shape))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, outp = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            PrintBatch(batch_x, input_width, 'input')
            PrintBatch(batch_y, label_width, 'label')
            outp = tf.reshape(outp, [batch_size, label_width, label_width, 3])
            outp = tf.split(0, batch_size, outp, name='split')
            labp = batch_y.reshape(batch_size, label_width, label_width, 3)
            labp = numpy.split(labp, batch_size, 0)
            for i in xrange(batch_size):
                arr = outp[i].eval()
                print(psnr(arr, labp[0]))
                assert arr.shape[0] == 1
                arr = arr.reshape(label_width, label_width, 3)
                misc.imsave('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/foo_pred'+str(i)+".jpeg", arr)
#                 #arr = misc.imresize(arr, (2, 128), mode='RGB')
#                 arr = Image.fromarray(arr, 'RGB')
#                 arr.save('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/foo_pred'+str(i)+".jpeg")
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))