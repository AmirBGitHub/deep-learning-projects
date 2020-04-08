# deep-learning-projects

# Guideline

After cloning/downloading the repository and executing command 'python main.py' in the first level directory, it will be able to generate all the results and plots used in the PDF report and print them out in a clear manner.

# Project 3

The purpose of this project is to implement a multiclass LR and a single hidden layer ANN classification algorithms on MNIST dataset and evaluate the performance on both MNIST and USPS handwritten digit data to identify them among 0, 1, 2, … , 9. The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems.

The database contains 60,000 training images and 10,000 testing images. The images are centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field. Each digit in the USPS dataset has 150 samples of “.png” file available for testing that need to be imported and converted into MNIST data format. The evaluation will be performed by calculating classification error rate in under the one-hot coding scheme.

# Project 4

The purpose of this project is to implement convolutional convolution neural network to determine whether the person in a portrait image is wearing glasses or not. The Celeb dataset was used which has more than 200k celebrity images in total. A publicly available convolutional neural network package from Tensorflow was used and trained on the Celeb images, hyperparameters were tuned and regularization was applied to improve the performance.

The CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with 202,599 celebrity images, each in color with an original definition of 178 x 218. Each image in the CelebA dataset is in “.jpg” format was imported and converted into 1D vector format. The evaluation will be performed by calculating classification error rate (𝐸 = 𝑁𝑤𝑟𝑜𝑛𝑔/𝑁𝑉) in under the one-hot coding scheme for “Eyeglasses” attribute from list_attr_celeba.txt file indicating whether the person in the picture is wearing glasses or not.
