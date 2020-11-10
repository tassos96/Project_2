import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Model, models
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from collections import deque

from utils import plotLoss, nextAction, nextLayer, addLayer
from input import readImages,readArgs

# gpu fix
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def encoder(input, convLayers): # input is of size 28 x 28 x 1, i.e. gray scale
    poolingLayers = 2
    firstLayer = True
    forbidPool = True # first layer cant be a pool
    # stack
    savedLayers = deque() # decoder mirrors encoder

    while convLayers > 0 or poolingLayers > 0: # keep count of remaining layers to be added
        # get next layer info from user
        layer_params, convLayers, poolingLayers = nextLayer(convLayers, poolingLayers, forbidPool)
        if layer_params['type'] == 'pool': # don't allow consecutive pooling layers
            forbidPool = True
        else:
            forbidPool = False

        # create the specified layer
        NN= addLayer(layer_params, input if firstLayer else NN)

        # save info for decoding
        savedLayers.append(layer_params)

        if firstLayer:
            firstLayer = False

    return NN, savedLayers

# 'mirrored' decoding sequence of layers
def decoder(NN, encdr_layers):
    while len(encdr_layers) > 0:
        # get next layer type based on encoder's layer
        nextLayer = encdr_layers.pop() # layer sequence is reverse with respect to encoder's
        # encoder has convolution layer, add deconvolution layer
        if nextLayer['type'] == 'conv':
            NN= addLayer(nextLayer, NN)
        # encoder has pooling layer, add upsampling layer
        else:
            NN= addLayer({'type':'upsample'}, NN)

    return Conv2D(1, (3,3), activation= 'sigmoid', padding= 'same')(NN) # final output layer


if __name__ == '__main__':
    while True:
        fileName, convNum, epochs, batchSize = readArgs()

        # read file with instances
        images, _, rows, cols = readImages(fileName)

        # scaling
        images = normalize(images, axis= 1)

        # 28 x 28 x 1 array per image, i.e. one channel for gray scale
        input_img = keras.Input(shape=(rows, cols, 1))

        # define structure of neural net
        NN, encdr_layers = encoder(input_img, convNum)
        NN = decoder(NN, encdr_layers)

        # build model
        autoencoder = Model(input_img, NN, name='N1')
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        print('\n\n~~~ Convolutional Neural Network Architecture ~~~\n')
        autoencoder.summary()

        # split into train set and validation set
        # the labels are the input images since we are training an autoencoder
        train, val, train_grnd, val_grnd = train_test_split(images, images, test_size=0.2, random_state=42)

        # training, training error and validation error returned for plotting
        errors = autoencoder.fit(train, train_grnd, batch_size=batchSize,
                                epochs=epochs, validation_data=(val, val_grnd))

        # ask user for the next action
        while True:
            doNext = nextAction()
            if doNext == 1 or doNext == 5: # repeat experiment or exit
                break
            elif doNext == 2: # plot losses
                plotLoss(errors.history)
            elif doNext == 3: # save weights of encoder
                autoencoder.save_weights(input('Enter path: '))
            elif doNext == 4: # save model and losses for research purposes
                models.save_model(autoencoder,input('Enter model path: '))
                np.save(input('Enter training history path: '),errors.history)

        if doNext == 5: # exit
            break

