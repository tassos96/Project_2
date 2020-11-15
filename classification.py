import tensorflow as tf
from tensorflow import keras
from keras import Model, models
from keras.layers import Dropout, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop

from classification_utils import FilePaths, getConvLayersIdx, \
    askDropout, readProb, actionToBool, get_setUntrainable, setTrainable, \
    nextAction
from input import getTrainParams, readVal, readImages, readLabels

# gpu fix
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# process the model created from first part and get the encoding part only
def encoder(autoencoder, input, dropAftrConv=False, dropProb=None):
    # get indexes of convolutional layers in order to place dropout layers right after
    if dropAftrConv:
        convLayersIdxs = getConvLayersIdx(autoencoder)

    # half of the model's layers participate in encoding, i.e. decoder mirrors encoder
    encLayers = len(autoencoder.layers) // 2

    for i in range(1,encLayers): # only iterate over encoder's layers
        NN = autoencoder.get_layer(index= i)(input if i == 1 else NN)
        # add dropout layer if needed (only if current layer was a convolution)
        if dropAftrConv and convLayersIdxs[i] == True:
            NN = Dropout(rate=dropProb)(NN)

    return NN, encLayers

def fullyConnected(NN, fcNodes, drop=False, dropProb=None):
    NN = Dense(fcNodes, activation='relu')(NN) # add fully connected layer
    if drop: # add dropout layer if needed (based on Hinton et al. 2012 paper)
        NN = Dropout(rate=dropProb)(NN)
    # 10 classes, i.e. numbers from range 0-9
    # softmax is used since we have multinomial classification
    return Dense(10, activation='softmax')(NN) # add output layer

if __name__=='__main__':
    paths = FilePaths()
    paths.setPaths()

    # read image files
    train, trn_img_n, rows, cols = readImages(paths.train_set_fname)
    test, tst_img_n, _, _ = readImages(paths.test_set_fname)

    # read label files
    train_y, trn_lbl_n = readLabels(paths.train_labels_fname)
    test_y, tst_lbl_n = readLabels(paths.test_labels_fname)

    assert trn_lbl_n == trn_img_n, 'Number of labels doesn\'t match number of images in training set'
    assert tst_lbl_n == tst_img_n, 'Number of labels doesn\'t match number of images in test set'

    # scaling i.e. normalization
    train /= 255.0
    test /= 255.0

    # split categorical attribute into 10 binary attributes, i.e. one-hot encoding
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # load autoencoder model
    autoencoder = models.load_model(paths.autoenc_model_fname, compile=False)

    while True:
        # get hyperparameters from user
        epochs, batchSize = getTrainParams()
        fullCnNodes = readVal(1, 'Number of nodes in fully connected layer: ')

        # dropout info
        dropAction = askDropout()
        dropAftrConv, dropAftrFC = actionToBool(dropAction)
        probConv, probFC = None, None
        if dropAftrConv:
            probConv = readProb('Dropout after Convolution layer probability: ')
        if dropAftrFC:
            probFC = readProb('Dropout after Fully connected layer probability: ')

        print(f'Prob Drop Conv: {probConv} \nProb Drop FC: {probFC}')

        # 28 x 28 x 1 array per image, i.e. one channel for gray scale
        input_img = keras.Input(shape=(rows, cols, 1))

        # process encoder part of autoencoder
        NN, encLayers = encoder(autoencoder, input_img, dropAftrConv, probConv)
        # add fully connected layer
        NN = fullyConnected(NN, fullCnNodes, dropAftrFC, probFC)

        # set weights of encoder to not be trained for the first step

        # build classification model
        classifier = Model(input_img, NN, name='N2')
        # get trained weights and don't train them
        get_setUntrainable(classifier, autoencoder, encLayers, dropAftrConv)

        classifier.compile(loss='categorical_crossentropy', optimizer=RMSprop())

        classifier.summary()

        # train only the fully connected layer

        # train the whole model

        # results


        # ask user for the next action
        doNext = nextAction()

        if doNext == 4: # exit
            break
