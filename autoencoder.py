import sys
import keras
from keras import layers, Model

from input import readImages

def encoder(inputs, filtersNum, filterSize, convNum):
    for i in range(4):  # 3 pooling layers
        for y in range(convNum): # add convolution layers before pooling
            neuralNet = Conv2D(filtersNum*(i+1),
                            filterSize,
                            activation= 'relu',
                            padding= 'same')(inputs if i == 0 and y == 0
                                                    else neuralNet)
            neuralNet = BatchNormalization()(neuralNet) # scaling layer
        neuralNet = MaxPooling2D(pool_size= 2)(neuralNet)

    return neuralNet

# 'mirrored' decoding sequence of layers
def decoder(neuralNet, filtersNum, filterSize, deconvNum):
    for i in range(4,0,-1):  # 3 upsampling layers
        neuralNet = UpSampling2D(size= 2)(neuralNet)
        for y in range(deconvNum): # add convolution layers after upsampling
            neuralNet = Conv2DTranspose(filtersNum*(i),
                                        filterSize,
                                        activation= 'relu',
                                        padding= 'same')(neuralNet)
            neuralNet = BatchNormalization()(neuralNet) # scaling layer


    neuralNet = Conv2DTranspose(1,
                                filterSize,
                                activation= 'linear',
                                padding= 'same')(neuralNet) # add an output layer
    return neuralNet



if __name__ == '__main__':
    x = input("enter: ")
    print(x)


