import keras
from keras import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.utils import normalize
from sklearn.model_selection import train_test_split

from utils import plotLoss, nextAction
from input import readImages,readArgs

def encoder(inputs, filtersNum, filterSize, convNum):
    # input is of size 28 x 28 x 1, i.e. gray scale
    for i in range(2):  # sequence of convolutions that extract features from increasing image subsets
        for y in range(convNum): # add convolution layers before pooling
            neuralNet = Conv2D(filtersNum*(y+1),
                            (filterSize,filterSize),
                            activation= 'relu',
                            padding= 'same')(inputs if i == 0 and y == 0
                                                    else neuralNet)

            neuralNet = BatchNormalization()(neuralNet) # scaling layer

        # pool to reduce number of parameters and keep important learned features
        neuralNet = MaxPooling2D((2,2), padding='valid')(neuralNet)

    return neuralNet

# 'mirrored' decoding sequence of layers
def decoder(neuralNet, filtersNum, filterSize, deconvNum):
    for i in range(2):  # 2 upsampling layers since we have 2 pooling layers in encoder
        for y in range(deconvNum, 0, -1): # add deconvolution layers before upsampling
            neuralNet = Conv2D(filtersNum*y,
                                        (filterSize,filterSize),
                                        activation= 'relu',
                                        padding= 'same')(neuralNet)

            neuralNet = BatchNormalization()(neuralNet) # scaling layer

        neuralNet = UpSampling2D((2,2))(neuralNet) # get back to initial dimensions

    neuralNet = Conv2D(1,
                        (filterSize,filterSize),
                        activation= 'sigmoid',
                        padding= 'same')(neuralNet) # 28 x 28 x 1 result, same size as input
    return neuralNet


if __name__ == '__main__':
    while True:
        fileName, filtersNum, filterSize, convNum, epochs, batchSize = readArgs()

        # read file with instances
        images, _, rows, cols = readImages(fileName)

        # scaling
        images = normalize(images, axis= 1)

        # 28 x 28 x 1 array per image
        input_img = keras.Input(shape=(rows, cols, 1))

        # define structure of neural net
        convNN = decoder(encoder(input_img, filtersNum, filterSize, convNum),
                        filtersNum, filterSize, convNum)

        # build model
        autoencoder = Model(input_img, convNN)
        autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        autoencoder.summary()

        # split into train set and validation set
        # the labels are the input images since we are training an autoencoder
        train, val, _, _ = train_test_split(images, images, test_size=0.2, random_state=42)

        # training, training error and validation error returned for plotting
        errors = autoencoder.fit(train, train, batch_size=batchSize, epochs=epochs, validation_data=(val, val))

        # ask user for the next action
        doNext = nextAction()
        if doNext == 2: # plot losses
            plotLoss(errors.history)
            if input('do you want to save the weights? [y|n]') == 'y':
                print(autoencoder.get_weights())
            break
        elif doNext == 3: # save weights of encoder
            print(autoencoder.get_weights())
            break
