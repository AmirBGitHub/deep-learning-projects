
# coding: utf-8

# In[5]:


# Project 3 Bayesian LR

print('UBitName = amirbagh')
print('personNumber = 50135018')

import numpy as np
import math
from sklearn.cluster import KMeans
from random import randint
import random
from PIL import Image 
import glob
from scipy import ndimage

# Read MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist_onehot = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


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

mnist_train_size = 5000
mnist_test_size = 1000

mnist_train_rand_idx = random.sample(range(1, mnist_onehot.train.images.shape[0]), mnist_train_size)
mnist_test_rand_idx = random.sample(range(1, mnist_onehot.test.images.shape[0]), mnist_test_size)

# comment/uncomment for MNIST or USPS test data
X_train = mnist_onehot.train.images[mnist_train_rand_idx]
#X_test = mnist_onehot.test.images[mnist_test_rand_idx]
X_test = uspsTest_images

y_v_train = mnist_onehot.train.labels[mnist_train_rand_idx]
#y_v_test = mnist_onehot.test.labels[mnist_test_rand_idx]
y_v_test =uspsLabels_oneHot

y_train = mnist.train.labels[mnist_train_rand_idx].reshape(mnist_train_size,1)
#y_test = mnist.test.labels[mnist_test_rand_idx].reshape(mnist_test_size,1)
y_test = uspsLabels

input_data_train = X_train
output_data_train = y_v_train

input_data_test = X_test
output_data_test = y_test

# Hyper parameters acquisition
def hyper_para_bayes_logistic(m_0, S_0, Theta, y):
    w_map = m_0
    S_N = np.linalg.inv(S_0)
    Theta = Theta.T
    for i in range(Theta.shape[0]):
        S_N = S_N + y[i]*(1-y[i])*np.matmul(Theta[i].T, Theta[i])  
    return w_map, S_N

def pred_bayes_logistic(w_map, S_N, theta):
    mu_a = np.dot(w_map.T, theta)
    var_a = np.dot(np.dot(theta.T, S_N), theta)
    kappa_var = (1 + math.pi*var_a/8)**(-0.5)
    x = kappa_var*mu_a
    return 1/(1 + np.exp(-x))

def training(Theta, y):
    w0 = np.random.normal(0, 1, Theta.shape[1])
    S0 = np.diag(np.random.normal(0, 1, Theta.shape[1]))
    # Theta n*m (n samples, m features), y n*1
    w_map, S_N = hyper_para_bayes_logistic(w0, S0, Theta, y)
    return w_map, S_N     

def predict(w_map, S_N, theta):
    return pred_bayes_logistic(w_map, S_N, theta)

# Multi-class classification    
def multiclass_sigmoid_logistic(Theta, Y):
    n_class = Y.shape[1]
    n_sample = Theta.shape[0]
    n_feature = Theta.shape[1]
    models_w = np.zeros((n_class,n_feature))
    models_s = np.zeros((n_class,n_feature,n_feature))
    for i in range(n_class):
        w_m , s_n = training(Theta, Y[:, i])
        models_w[i] = w_m
        models_s[i] = s_n
    return models_w, models_s

# Prediction
def pred_multiclass_sigmoid_logistic(theta, Theta, Y):
    models_w, models_s = multiclass_sigmoid_logistic(Theta, Y)
    props = []
    for i in range(len(models_w)):
        props.append(predict(models_w[i], models_s[i], theta))
    return np.nanmax(props), np.nanargmax(props), props

idx_v = []
max_props_v = []
max_props_idx_v = []
props_v = []
    
for t in range(0, 100):
    idx = randint(1, np.shape(input_data_test)[0]-1)
    max_props, max_props_idx, props = pred_multiclass_sigmoid_logistic(theta=input_data_test[idx],
                                                                       Theta=input_data_train, 
                                                                       Y=output_data_train)
    idx_v.append(idx)
    max_props_v.append(max_props)
    max_props_idx_v.append(max_props_idx)
    props_v.append(props)

true_testlabels = output_data_test[idx_v]
pred_testlabels = max_props_idx_v

from sklearn.metrics import accuracy_score
print('test accuracy:', accuracy_score(true_testlabels, pred_testlabels))

