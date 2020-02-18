# Importing required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Load the data from the keras fashion_mnist dataset
# Reference : https://keras.io/datasets/
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Number of Images along with the pixel of each image in training dataset are: "+str(x_train.shape))
print("Number of Labels in training dataset are: "+str(y_train.shape))
print("Number of Images along with the pixel of each image in testing dataset are:"+str(x_test.shape))
print("Number of Labels in testing dataset are: "+str(y_test.shape))

#Class labels do not come with dataset and hence we code them here
# Reference : https://keras.io/datasets/
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Plot any random training image
plt.figure()
a = randint(1, len(x_train))
plt.imshow(x_train[a])
plt.show()

#plot any random testing image
plt.figure()
b = randint(1, len(x_test))
plt.imshow(x_test[b])
plt.show()

print(x_train[a])

# From the result of above print statement we can see that an image is read with the values ranging from 0-255
# Scaling all testing and training images to range of 0-1
x_train = x_train/255.0
x_test = x_test/255.0

print(x_train[a])

# Number of layers in neural network are designed
network = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #transforms data from 2-d to 1-d
    keras.layers.Dense(128, activation=tf.nn.relu), #This layer has 128 nodes with relu as activation function
    keras.layers.Dense(10, activation=tf.nn.softmax) #converts all values to within range 0-1
])

# Compiling the network
network.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the network
network.fit(x_train, y_train, epochs=10)

predictions = network.predict(x_test)

# Maximum value in the corresponding image array
te = randint(0, len(x_test))
np.argmax(predictions[te])

y_test[te]

img1 = randint(0, len(x_test))
img = x_test[img1]
print(img.shape)

#Reference https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
img = (np.expand_dims(img,axis=0))
print(img.shape)

pred1 = network.predict(img)
pred2 = pred1[0]
print(pred2)

ma = np.argmax(pred1[0])
print(ma)

##Verification to check whether all probabilities sum to 1
val1 = 0
for val in pred2:
    val1 = val1 + val
print("Sum of all predictions = "+str(val1))

#Calculating accuracy percentage
accuracy = pred2[ma] * 100

len = np.arange(10)
plt.figure()
plt.imshow(x_test[img1])
plt.show()

plt.bar(len,pred2)
plt.xticks(len, class_labels, rotation=90)
plt.ylabel('Probability of prediction')
plt.xlabel('Type of Image')
plt.title('Image recognition')
plt.show()

# Output
na = y_test[img1]
print("Testing image fed to the network is: "+class_labels[na])
print("Image predicted is: "+class_labels[ma])
print("Accuracy of prediction is: "+str(accuracy)+"%")
