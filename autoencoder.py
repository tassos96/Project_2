import keras
from keras import Model
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.utils import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from input import readImages,readArgs

plt.style.use('ggplot')

def encoder(inputs, filtersNum, filterSize, convNum):
    for i in range(2):  # 2 pooling layers
        for y in range(convNum): # add convolution layers before pooling
            neuralNet = Conv2D(filtersNum*(y+1),
                            (filterSize,filterSize),
                            activation= 'relu',
                            padding= 'same')(inputs if i == 0 and y == 0
                                                    else neuralNet)
            neuralNet = BatchNormalization()(neuralNet) # scaling layer

        neuralNet = MaxPooling2D((2,2), padding='same')(neuralNet)

    return neuralNet

# 'mirrored' decoding sequence of layers
def decoder(neuralNet, filtersNum, filterSize, deconvNum):
    for i in range(2):  # 2 upsampling layers
        for y in range(deconvNum, 0, -1): # add deconvolution layers after upsampling
            neuralNet = Conv2D(filtersNum*y,
                                        (filterSize,filterSize),
                                        activation= 'relu',
                                        padding= 'same' if i == 1 and y == 1
                                                         else 'same')(neuralNet)
            neuralNet = BatchNormalization()(neuralNet) # scaling layer

        neuralNet = UpSampling2D((2,2))(neuralNet)

    neuralNet = Conv2D(1,
                        (filterSize,filterSize),
                        activation= 'linear',
                        padding= 'same')(neuralNet) # add an output layer
    return neuralNet


if __name__ == '__main__':
    fileName, filtersNum, filterSize, convNum, epochs, batchSize = readArgs()

    # read file with instances
    images, _, rows, cols = readImages(fileName)

    # scaling
    images = normalize(images, axis= 1)

    input_img = keras.Input(shape=(rows, cols, 1))
    autoencoder = Model(input_img, decoder(encoder(input_img, filtersNum, filterSize, convNum),
                                        filtersNum,
                                        filterSize,
                                        convNum)
                        )
    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    autoencoder.summary()
    # the labels are the input images since we are training an autoencoder
    train, val, _, _ = train_test_split(images, images, test_size=0.2, random_state=42)

    errors = autoencoder.fit(train, train, batch_size=batchSize, epochs=epochs, validation_data=(val, val))
    print(errors.history)

    #Plotting the validation and training errors
    x_axis = range(len(errors.history['loss']))
    plt.plot(x_axis, errors.history['val_loss'], label='val', linestyle='--')
    plt.plot(x_axis, errors.history['loss'], label='train')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

