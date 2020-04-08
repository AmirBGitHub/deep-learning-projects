
# coding: utf-8

# In[1]:


# Project 4

print('UBitName = amirbagh')
print('personNumber = 50135018')

import numpy as np                                            
from PIL import Image 
import glob
from scipy import ndimage
import scipy

# setting the parameters
celebaImages_1d = []
new_width  = 32
new_height = 40
sample_size = 2000
itr = 200
tr_itr = 10
batch_size = 10
n_classes = 2
train_percent = 0.9 
val_percent = 0.05
test_percent = 0.05

# read the CelebA image data
for fname in glob.glob('data/*.jpg'):
    img = scipy.ndimage.imread(fname, flatten=True, mode=None)
    img_resize = scipy.misc.imresize(img.astype('float32'), [new_width, new_height])
    celebaImages_1d.append(img_resize.ravel())
    
celebaImages = np.asarray(celebaImages_1d)  

# reading the labels from "list_attr_celeba.txt" file    
f = open("data/list_attr_celeba.txt", "r")
list_attr_celeba = []
for line in f:
    list_attr_celeba.append(line)
eyeglass_label = []    
list_attr_celeba = list_attr_celeba[0:sample_size+2]
for i in range(2,np.size(list_attr_celeba)):
    attr = list_attr_celeba[i].split()
    eyeglass_label.append(int((int(attr[16])+1)/2))
    
celebaLabels = eyeglass_label
celebaLabels_oneHot = np.zeros((np.size(list_attr_celeba)-2, 2))
celebaLabels_oneHot[np.arange(np.size(list_attr_celeba)-2), celebaLabels] = 1  

def dataset_partition(train_percent, val_percent, test_percent, data, indexes):
    data_lngth = np.shape(data)[0]
    train_data = data[indexes[0:round(train_percent * data_lngth)],:]
    val_data = data[indexes[round(train_percent * data_lngth):round((train_percent + val_percent) * data_lngth)],:]
    test_data = data[indexes[round((train_percent + val_percent) * data_lngth):data_lngth],:]
    return (train_data, val_data, test_data)

# Augment training data
def expend_training_data(images, labels, new_width, new_height):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,np.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = np.median(x) # this is regarded as background's value        
        image = np.reshape(x, (-1, new_width))

        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(np.reshape(new_img_, new_width*new_height))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = np.concatenate((expanded_images, expanded_labels), axis=1)
    np.random.shuffle(expanded_train_total_data)

    trainData_aug = expanded_train_total_data[:,0:new_width*new_height]
    trainLabel_aug = expanded_train_total_data[:,new_width*new_height:expanded_train_total_data.shape[1]]
    expanded_train_total_data_seg = [trainData_aug, trainLabel_aug]
    
    return expanded_train_total_data_seg

# selecting the data
data_choose_in = celebaImages
data_choose_out = celebaLabels_oneHot

# shuffling the data
data_index = list(range(0, np.shape(data_choose_in)[0]))
np.random.shuffle(data_index)

# data partition
data_partition_in = dataset_partition(train_percent=train_percent, val_percent=val_percent, test_percent=test_percent, data=data_choose_in, indexes=data_index)
data_partition_out = dataset_partition(train_percent=train_percent, val_percent=val_percent, test_percent=test_percent, data=data_choose_out, indexes=data_index)

# test data assignment
testData = data_partition_in[1]
testLabel = data_partition_out[1]

# train data assignment/augmentation
celeba_train = [data_partition_in[0], data_partition_out[0]]
celeba_train_aug = expend_training_data(celeba_train[0], celeba_train[1], new_width, new_height)

# comment/uncomment for training data augmentation
#trainData = celeba_train[0]
#trainLabel = celeba_train[1]
trainData = celeba_train_aug[0]
trainLabel = celeba_train_aug[1]


# Convolutional Neural Network

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, new_width*new_height])  # input
W = tf.Variable(tf.zeros([new_width*new_height, 2]))          # Parameter Variable
b = tf.Variable(tf.zeros([n_classes]))               
y = tf.nn.softmax(tf.matmul(x, W) + b)       # Softmax regression model

y_ = tf.placeholder(tf.float32, [None, n_classes])   # add a new placeholder

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
x_image = tf.reshape(x, [-1, new_width, new_height, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Third Convolutional Layer
W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) +
b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# Densely Connected Layer
W_fc1 = weight_variable([(new_width//8) * (new_height//8) * 128, 2048])
b_fc1 = bias_variable([2048])
h_pool3_flat = tf.reshape(h_pool3, [-1, (new_width//8)*(new_height//8)*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Readout Layer
W_fc2 = weight_variable([2048, n_classes])
b_fc2 = bias_variable([n_classes])
# comment/uncomment for dropout
#y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# Train and Evaluate the Model
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    test_err = []
    for i in range(itr-1):
        batch = [trainData[batch_size*(i//tr_itr):batch_size*(i//tr_itr+1)], trainLabel[batch_size*(i//tr_itr):batch_size*(i//tr_itr+1)]]
        if i % tr_itr == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        test_err.append(1-accuracy.eval(feed_dict={
            x: testData, y_: testLabel, keep_prob: 1}))
    
    print('testing error rate %g' % np.round(np.mean(test_err),2))
        
        

    

