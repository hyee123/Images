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
(X_train, y_train), (X_test, y_test) = mnist.load_data()

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
model = Sequential()

# Input layer 
# Convolution2D(#convolutionFilters, #rows, #cols for each kernel)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28)))

# More layers
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # reduce the number of parameters
model.add(Dropout(0.25)) # regularize model to prevent overfitting

model.add(Flatten()) # make 1 dimensional
model.add(Dense(128, activation='relu')) # first para is output size
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model on training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)
model.save('new_model.h5')

score = model.evaluate(X_test, Y_test, verbose=0)
print 'Score: ' + score
