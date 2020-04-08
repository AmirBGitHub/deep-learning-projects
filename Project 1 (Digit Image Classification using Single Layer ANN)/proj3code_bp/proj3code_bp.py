
# coding: utf-8

# In[1]:


# Project 3 Back Propagation

print('UBitName = amirbagh')
print('personNumber = 50135018')

# test size 
test_size=0.4

from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt 
from PIL import Image 
import glob
from scipy import ndimage
#print(digits.data.shape)
#plt.gray() 
#plt.matshow(digits.images[1]) 
#plt.show()


# Normalize Data
from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

# Split Data
from sklearn.model_selection import train_test_split
y = digits.target
X_train, X_test_mnist, y_train, y_test_mnist = train_test_split(X, y, test_size=test_size)

# Convert to one-hot representation
import numpy as np
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
y_v_train = convert_y_to_vect(y_train)
y_v_test_mnist = convert_y_to_vect(y_test_mnist)


# read the USPS data
uspsTest_images_1d = []
new_width  = 8
new_height = 8
for filename in glob.glob('USPSTest/*.png'): 
    im=Image.open(filename)
    im_resize = im.resize((new_width, new_height), Image.ANTIALIAS)
    im_resize_smooth = ndimage.generic_filter(im_resize, np.nanmean, size=3, mode='constant', cval=np.NaN)
    im_1d = np.array(im_resize_smooth, dtype='f').ravel()/-255+1
    uspsTest_images_1d.append(im_1d)
    
X_test_usps = StandardScaler().fit_transform(np.reshape(uspsTest_images_1d, 
    (int(np.size(uspsTest_images_1d)/(new_width*new_height)), (new_width*new_height))))


uspsDigits = list(range(10))
uspsDigits.reverse()
y_test_usps = np.repeat(uspsDigits, 150)
y_v_test_usps = np.zeros((1500, 10))
y_v_test_usps[np.arange(1500), y_test_usps] = 1

# comment/uncomment for using USPS or MNIST data
#X_test = X_test_usps
#y_v_test =y_v_test_usps
#y_test = y_test_usps

X_test = X_test_mnist
y_v_test =y_v_test_mnist
y_test = y_test_mnist


# Creating the Neural Network (one hidden layer)
nn_structure = [64, 30, 10]

# Sigmoid activation function
def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))

# Randomly initialise the weights
import numpy.random as r
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

# Initialize the weight and bias values
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

# Feed forward pass 
def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # if it is the first layer, then the input into the weights is x, otherwise, 
        # it is the output from the last layer
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
    return h, z

# calculate the output layer delta δ(nl) and hidden layer delta δ(l)
def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

# Train NN with backpropagation of errors
def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)
            # loop from nl-1 to 1 backpropagating the errors
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # perform the gradient descent step for the weights in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # complete the average cost calculation
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

# Run training and plot
W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

# Testing the NN
def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])
    return y

from sklearn.metrics import accuracy_score
y_pred = predict_y(W, b, X_test, 3)
print('test accuracy:', accuracy_score(y_test, y_pred))

