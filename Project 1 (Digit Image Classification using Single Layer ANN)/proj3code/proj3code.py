
# coding: utf-8

# In[2]:


# Project 3

print('UBitName = amirbagh')
print('personNumber = 50135018')

import numpy as np                                            
from PIL import Image 
import glob
from scipy import ndimage

# download and read the MNIST data
from tensorflow.examples.tutorials.mnist import input_data    
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# read the USPS data
uspsTest_images_1d = []
new_width  = 28
new_height = 28
for filename in glob.glob('USPSTest/*.png'): 
    im=Image.open(filename)
    im_resize = im.resize((new_width, new_height), Image.ANTIALIAS)
    im_resize_smooth = ndimage.generic_filter(im_resize, np.nanmean, size=3, mode='constant', cval=np.NaN)
    im_1d = np.array(im_resize_smooth, dtype='f').ravel()/-255+1
    uspsTest_images_1d.append(im_1d)
    
uspsTest_images = np.reshape(uspsTest_images_1d, 
    (int(np.size(uspsTest_images_1d)/(new_width*new_height)), (new_width*new_height)))
            
uspsDigits = list(range(10))
uspsDigits.reverse()
uspsLabels = np.repeat(uspsDigits, 150)
uspsLabels_oneHot = np.zeros((1500, 10))
uspsLabels_oneHot[np.arange(1500), uspsLabels] = 1

# comment/uncomment for MNIST or USPS test data
trainData = mnist.train.images
#testData = mnist.test.images
testData = uspsTest_images
trainLabel = mnist.train.labels
#testLabel = mnist.test.labels
testLabel = uspsLabels_oneHot


import tensorflow as tf

# Single Hidden Layer Neral Network

x = tf.placeholder(tf.float32, [None, 784])  # input
W = tf.Variable(tf.zeros([784, 10]))         # Parameter Variable
b = tf.Variable(tf.zeros([10]))               
y = tf.nn.softmax(tf.matmul(x, W) + b)       # Softmax regression model

y_ = tf.placeholder(tf.float32, [None, 10])  # add a new placeholder

cross_entropy = tf.reduce_mean(              # cross-entropy function
                -tf.reduce_sum(y_ * tf.log(y),
                reduction_indices=[1]))

# optimize variables and reduce the loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) 

# launch the model in an InteractiveSession
sess = tf.InteractiveSession()

# create an operation to initialize the variables 
tf.global_variables_initializer().run()
# we'll run the training step 1000 times
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# to determine what fraction are correct, we
# cast to floating point numbers and then
# take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# finally, we ask for our accuracy on our
# test data
print(sess.run(accuracy, feed_dict={x: testData, y_: testLabel}))



# Convolutional Neural Network

# Weight Initialization 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')

# First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: testData, y_: testLabel, keep_prob: 1.0}))
        
        
        

    

