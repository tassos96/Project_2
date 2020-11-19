import tensorflow as tf
from tensorflow import keras
from keras import Model, models
from keras.layers import Dropout, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from classification_utils import *
from input import getTrainParams, readVal, readImages, readLabels
from utils import plotLoss

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
    NN = Flatten()(NN)
    NN = Dense(fcNodes, activation='relu')(NN) # add fully connected layer
    if drop: # add dropout layer if needed (based on Hinton et al. 2012 paper)
        NN = Dropout(rate=dropProb)(NN)
    # 10 classes, i.e. numbers from range 0-9
    # softmax is used since we have multinomial classification
    return Dense(10, activation='softmax')(NN) # add output layer

if __name__=='__main__':
    save = [] # for every experiment keep hyperparameters and losses for plotting
    # 'save' list contains dicts with following attributes:
    # number of nodes of fully connected layer
    # batch size
    # epochs
    # last epoch of first phase
    # training loss & validation loss
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
    test = normalize(test)
    train = normalize(train)

    # split categorical attribute into 10 binary attributes, i.e. one-hot encoding
    train_y = to_categorical(train_y)

    # load autoencoder model
    autoencoder = models.load_model(paths.autoenc_model_fname, compile=False)

    while True:
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
        get_setUntrainable(classifier, autoencoder, encLayers)

        classifier.compile(loss='categorical_crossentropy', optimizer=Adam())

        classifier.summary()

        # split into train set and validation set
        train_curr, val, train_curr_y, val_y = train_test_split(train, train_y, test_size=0.2, random_state=42)

        print('Training weights of fully connected layer...')
        # get hyperparameters about training process from user
        epochsPhase1, batchSize = getTrainParams()
        # train only the fully connected layer
        metricsPhase1= classifier.fit(train_curr, train_curr_y, batch_size=batchSize,\
                                epochs=epochsPhase1, validation_data=(val,val_y))

        print('Training weights of all layers...')
        epochsPhase2 = readVal(1, 'Epochs: ')
        # make encoder's weights trainable
        setTrainable(classifier, encLayers)
        # warm start
        classifier.compile(loss='categorical_crossentropy', optimizer=Adam())
        # retrain all the weights
        metricsPhase2= classifier.fit(train_curr, train_curr_y, batch_size=batchSize,\
                                epochs=epochsPhase2, validation_data=(val,val_y))

        errors = concatErrorHistory(metricsPhase1,metricsPhase2)

        # for plotting
        saveInfo(batchSize, fullCnNodes, epochsPhase1+epochsPhase2, errors['loss'], errors['val_loss'], epochsPhase1, save)

        while True:
            # ask user for the next action
            doNext = nextAction()

            if doNext == 1 or doNext == 5:
                break
            elif doNext == 2: # plot losses
                plotLoss(errors, loss_fn='Cross Entropy', ep_first_phase=epochsPhase1)
            elif doNext == 3:
                test_pred = classifier.predict(test) # predict classes of images in test set

                test_pred = np.argmax(test_pred,axis=1) # get class with highest softmax probability

                target_names = [f'Class {i}' for i in range(10)] # make output more clear
                # output table with metrics
                print(classification_report(test_y, test_pred, target_names=target_names))

                correct = np.where(test_y==test_pred)[0]
                incorrect = np.where(test_y!=test_pred)[0]
                print(f'Predicted {len(correct)} images correctly and {len(incorrect)} images incorrectly\n')

                corrImgCount = readVal(0, 'How many correctly predicted images to show?: ')
                showImgs(corrImgCount, correct, test, test_pred, test_y)

                falseImgCount = readVal(0, 'How many incorrectly predicted images to show?: ')
                showImgs(falseImgCount, incorrect, test, test_pred, test_y)
            elif doNext == 4:
                if len(save) == 1:
                    print('Not enough experiments done, use option 2 instead.')
                    continue
                plotAll(save)

        if doNext == 5: # exit
            break





