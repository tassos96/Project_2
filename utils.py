from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D
import matplotlib.pyplot as plt

# fancy graphics
plt.style.use('ggplot')

# ask user for next layer type and layer parameters
def nextLayer(convLayers, poolingLayers):
    layer = {}
    prompt ="""
Please specify next layer, you have
{} Convolutional layers and
{} Pooling layers left.
Enter [conv|pool]:
""".format(convLayers, poolingLayers)

    layer['type'] = input(prompt)
    if poolingLayers > 0 and convLayers > 0:
        while layer['type'] not in ('conv','pool'):
            layer['type'] = input('Wrong input, re-enter: ')
    elif poolingLayers == 0:
        while layer['type'] != 'conv':
            layer['type'] = input('You can only add a Convolutional layer, re-enter: ')
    elif convLayers == 0:
        while layer['type'] != 'pool':
            layer['type'] = input('You can only add a Pooling layer, re-enter: ')

    if layer['type'] == 'conv':
        convLayers -= 1
        while True:
            try:
                layer['filter_num'] = int(input('Specify number of filters: '))
                break
            except:
                print('Input error!')

        while True:
            try:
                layer['filter_size'] = int(input('Specify kernel size: '))
                break
            except:
                print('Input error!')
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
    plt.plot(x_axis, losses['val_loss'], label='Validation loss', linestyle='--')
    plt.plot(x_axis, losses['loss'], label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def nextAction():
    prompt = """
Enter one of 1, 2, 3:
\t1) repeat process
\t2) plot training and validation loss over epochs
\t3) save encoder weights
"""
    action = input(prompt)
    while action not in ['1','2','3']:
        action = input("Wrong input, enter again: ")

    return int(action)

