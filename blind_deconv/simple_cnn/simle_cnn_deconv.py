
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

# Import MNIST data
import input_data
from tensorflow.models.image.mnist.convolutional import BATCH_SIZE
mnist = input_data.read_data_sets()

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 10
display_step = 10

# Network Parameters
input_width = 64
label_width = 32
n_input = input_width*input_width*3 # MNIST data input (img shape: 28*28)
n_classes = label_width*label_width*3 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, input_width, input_width, 3])
y = tf.placeholder(tf.float32, [batch_size, label_width * label_width * 3])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, keep_prob):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    print ('x ' + str(x.get_shape()))
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print ('conv1 ' + str(conv1.get_shape()))
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print ('conv1 ' + str(conv1.get_shape()))

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print ('conv2 ' + str(conv2.get_shape()))
    # Max Pooling (down-sampling)
    #conv2 = maxpool2d(conv2, k=2)
    print ('conv2 ' + str(conv2.get_shape()))

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.nn.conv2d_transpose(conv2, weights['wd1'], \
                                 [batch_size, label_width, label_width, 3], strides=[1, 1, 1, 1])
    print ('deconv ' + str(fc1.get_shape()))
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return fc1

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([2, 2, 3, 32], mean=1.0, stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([2, 2, 32, 64], mean=1.0, stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32, 32, 3, 64], mean=1.0, stddev=0.1)),
#     # 1024 inputs, 10 outputs (class prediction)
#     'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
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
cost = tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred2, y)))
print ('Cost shape: ' + str(cost.get_shape()))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
print ('Adding evaluation model')
accuracy = tf.div(tf.sqrt(tf.mul(tf.reduce_mean(tf.nn.l2_loss(tf.sub(pred2, y))), 2.0)), batch_size)

# Initializing the variables
print ('Initializing variables')
init = tf.initialize_all_variables()

# Launch the graph
print ('Launching session')
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #print ('Processing ' + str(step))
        batch_x, batch_y = mnist.train.next_batch(batch_size)
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
            outp = tf.reshape(outp, [batch_size, label_width, label_width, 3])
            outp = tf.split(0, batch_size, outp, name='split')
            for i in xrange(batch_size):
                arr = outp[i].eval()
                assert arr.shape[0] == 1
                arr = arr.reshape(label_width, label_width, 3)
                arr = misc.imresize(arr, (input_width, input_width), mode='RGB')
                arr = Image.fromarray(arr, 'RGB')
                arr.save('/Users/vaibhavg/Desktop/Stanford/AI/Project/Dataset/training/output/foo3'+str(i)+".jpeg")
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
