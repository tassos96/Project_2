from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
import matplotlib.pyplot as plt
from input import readVal

# fancy graphics
plt.style.use('seaborn')
plt.tight_layout()
# plt.xkcd()

# ask user for next layer type and layer parameters
def nextLayer(convLayers, poolingLayers, forbidPool):
    layer = {}
    prompt ="""
Please specify next layer,
Convolutional layers left: {}
Pooling layers left: {}
Enter [conv|pool]: """.format(convLayers, poolingLayers)

    layer['type'] = input(prompt)
    # don't allow inefficient architectures, e.g. all convolutions first and then all pools
    if poolingLayers == 0 or forbidPool: # no pools left or previous layer is pool or this is first layer
        while layer['type'] != 'conv':
            layer['type'] = input('You can only add a Convolutional layer, re-enter: ')
    elif convLayers == 0 or (poolingLayers == 2 and convLayers == 1): # second operand prevents dead-end
        while layer['type'] != 'pool':
            layer['type'] = input('You can only add a Pooling layer, re-enter: ')
    else: # can add any of the two, i.e. convolution or pool
        while layer['type'] not in ('conv','pool'):
            layer['type'] = input('Wrong input, re-enter: ')

    if layer['type'] == 'conv':
        convLayers -= 1
        layer['filter_num'] = readVal(1, 'Specify number of filters: ')
        layer['filter_size'] = readVal(1, 'Specify kernel size: ')
    else:
        poolingLayers -= 1

    return layer, convLayers, poolingLayers

# add layer of neurons specified by user
def addLayer(layer_params, input): # layer_params is a dict with layer-specific information
    if layer_params['type'] == "conv": # layer that extracts features from subset of input
        NN= Conv2D(layer_params['filter_num'], layer_params['filter_size'],\
            activation= 'relu', padding= 'same')(input)
        return BatchNormalization()(NN) # scaling
    elif layer_params['type'] == "pool":
        return MaxPooling2D((2,2))(input) # reduce number of parameters and keep important learned features
    elif layer_params['type'] == "upsample":
        return UpSampling2D((2,2))(input) # get back to initial dimensions

def plotLoss(losses):
    #Plotting the validation and training errors
    x_axis = range(len(losses['loss']))
    plt.plot(x_axis, losses['loss'], label='Training loss', c='orange')
    plt.plot(x_axis, losses['val_loss'], label='Validation loss', linestyle='-.', c='brown', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    # plt.xticks(x_axis)
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def nextAction():
    prompt = """
Enter one of 1, 2, 3, 4, 5:
\t1) repeat process
\t2) plot training and validation loss over epochs
\t3) save encoder weights
\t4) save model and losses
\t5) exit
"""
    action = input(prompt)
    while action not in {'1','2','3','4','5'}:
        action = input("Wrong input, enter again: ")

    return int(action)

