import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Conv2D
from keras.optimizers import Adam, RMSprop
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from collections import deque

from utils import plotLoss, nextAction, nextLayer, addLayer
from input import readImages,readArgs

# gpu fix
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def encoder(input, convLayers):
    # input is of size 28 x 28 x 1, i.e. gray scale
    poolingLayers = 2
    firstLayer = True
    # queue        stack
    decoderLayers, decoderConvParams = deque(), deque() # decoder mirrors encoder

    while convLayers > 0 or poolingLayers > 0: # keep count of remaining layers to be added
        # get next layer info from user
        layer_params, convLayers, poolingLayers = nextLayer(convLayers, poolingLayers)
        # create the specified layer
        NN= addLayer(layer_params, input if firstLayer else NN)

        # save info for decoding
        decoderLayers.append(layer_params['type'])
        if layer_params['type'] == 'conv': # decoder will have same filter amount and size, so keep this info
            decoderConvParams.append({'filter_num':layer_params['filter_num'],'filter_size':layer_params['filter_size']})

        if firstLayer:
            firstLayer = False

    return NN, decoderLayers, decoderConvParams

# 'mirrored' decoding sequence of layers
def decoder(NN, layer_types, convParams):
    while len(layer_types) > 0:
        # get next layer type based on encoder's layer
        nextLyr = layer_types.popleft() # layer sequence is same as encoder
        # encoder has convolution layer, add deconvolution layer
        if nextLyr == 'conv':
            params = convParams.pop() # filter params are inverse with respect to encoder
            params['type'] = 'conv'
            NN= addLayer(params, NN)
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

        # 28 x 28 x 1 array per image
        input_img = keras.Input(shape=(rows, cols, 1))

        # define structure of neural net
        NN, layer_types, convParams = encoder(input_img, convNum)
        NN = decoder(NN, layer_types, convParams)

        # build model
        autoencoder = Model(input_img, NN)
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        autoencoder.summary()

        # split into train set and validation set
        # the labels are the input images since we are training an autoencoder
        train, val, train_grnd, val_grnd = train_test_split(images, images, test_size=0.2, random_state=42)

        # training, training error and validation error returned for plotting
        errors = autoencoder.fit(train, train_grnd, batch_size=batchSize,
                                epochs=epochs, validation_data=(val, val_grnd))

        # ask user for the next action
        doNext = nextAction()
        if doNext == 2: # plot losses
            plotLoss(errors.history)
            if input('do you want to save the weights? [y|*]: ') == 'y':
                autoencoder.save_weights(input('Enter path: '))
            break
        elif doNext == 3: # save weights of encoder
            autoencoder.save_weights(input('Enter path: '))
            break

