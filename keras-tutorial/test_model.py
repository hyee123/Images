import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt # visualization
np.random.seed(123) # for reproducibility

from keras import backend as K
K.set_image_dim_ordering('th')

# Sequential model type (linear stack of layers for feed-forward CNN)
from keras.models import Sequential
# Core layers from keras, can be used in any neural net
from keras.layers import Dense, Dropout, Activation, Flatten
# CNN layers
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils # Utilities for transforming data

# Load image data from MNIST
from keras.datasets import mnist
(X_train, y_train), (X_test_ori, y_test) = mnist.load_data()

X_test = X_test_ori
# Declare depth of 1 (greyscale)
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# Change to float in range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Split the y set to 10 classes
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define model architecture
from keras.models import load_model
model = load_model('new_model.h5')

num_predicts = 20
predictions = model.predict(X_test[:num_predicts])

for i in range(0, num_predicts):
    maxVal = max(predictions[i])
    prediction = 0
    for j in range(0, 10):
        if maxVal == predictions[i][j]:
            prediction = j
    plt.title('Prediction: ' + str(prediction))
    plt.imshow(X_test_ori[i])
    plt.show()
#score = model.evaluate(X_test, Y_test, verbose=0)
#print model.metrics_names
#print 'Score: ' + str(score)
