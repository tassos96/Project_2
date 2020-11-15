import sys


class FilePaths:

    def __init__(self):
        self.train_set_fname = None
        self.train_labels_fname = None
        self.test_set_fname = None
        self.test_labels_fname = None
        self.autoenc_model_fname = None


    # read command line arguments
    def setPaths(self):
        for i,arg in enumerate(sys.argv):
            if arg == '-d':
                self.train_set_fname = sys.argv[i+1]
            elif arg == '-dl':
                self.train_labels_fname = sys.argv[i+1]
            elif arg == '-t':
                self.test_set_fname = sys.argv[i+1]
            elif arg == '-tl':
                self.test_labels_fname = sys.argv[i+1]
            elif arg == '-model':
                self.autoenc_model_fname = sys.argv[i+1]

        self.askUser()

    def isNone(self, val):
        if val == None:
            return True

        return False

    # get values for unspecified args on exec time
    def askUser(self):
        if self.isNone(self.train_set_fname):
            self.train_set_fname = input('Training set path: ')

        if self.isNone(self.train_labels_fname):
            self.train_labels_fname = input('Training labels path: ')

        if self.isNone(self.test_set_fname):
            self.test_set_fname = input('Test set path: ')

        if self.isNone(self.test_labels_fname):
            self.test_labels_fname = input('Test labels path: ')

        if self.isNone(self.autoenc_model_fname):
            self.autoenc_model_fname = input('Autoencoder model path: ')

'''
~~~~~~~~~ END OF CLASS
'''

# used for dropout, i.e. locate convolutional layers
def getConvLayersIdx(autoencoder):
    convLayers = []

    for layer in autoencoder.layers:
        if 'conv2d' in layer.name:
            convLayers.append(True)
        else:
            convLayers.append(False)

    return convLayers

'''
ask user if he wants dropout layers to be added
in order to prevent overfitting
'''
def askDropout():
    prompt = """
Enter one of 1, 2, 3, 4:
\t1) no dropout layers
\t2) dropout layers after convolution layers
\t3) dropout layer after fully connected layer
\t4) both 2 and 3
"""
    action = input(prompt)
    while action not in {'1','2','3','4'}:
        action = input("Wrong input, enter again: ")

    return int(action)

'''
if dropout layers to be added, ask for the probability
of a neuron getting dropped out
'''
def readProb(mssg):
    while True:
        try:
            prob = float(input(mssg))
            if prob >= 1.0 or prob <= 0.0:
                raise Exception
            break
        except:
            print('Input error!')

    return prob

'''
convert user input into method arguments
'''
def actionToBool(dropoutAction):
    # returns two bools (dropout after convolution, >> after full conn layer)
    if dropoutAction == 1:
        return False, False
    elif dropoutAction == 2:
        return True, False
    elif dropoutAction == 3:
        return False, True
    else:
        return True, True


'''
load weights of encoder from pre-trained autoencoder(first task)
and make them non trainable
'''
def get_setUntrainable(classifier, autoencoder, encLayers):
    counter = 0
    for i in range(encLayers):
        class_idx = i + counter
        if 'dropout' in classifier.layers[class_idx].name: # skip dropout layer
            class_idx += 1
            counter += 1

        # get trained weights from autoencoder
        classifier.layers[class_idx].set_weights(autoencoder.layers[i].get_weights())
        # don't train the weights in first training phase
        classifier.layers[class_idx].trainable = False


'''
make weights of encoder trainable
'''
def setTrainable(classifier, encLayers):
    counter = 0
    for i in range(encLayers):
        class_idx = i + counter
        if 'dropout' in classifier.layers[class_idx].name: # skip dropout layer
            class_idx += 1
            counter += 1

        # train the weights in second training phase
        classifier.layers[class_idx].trainable = True


def nextAction():
    prompt = """
Enter one of 1, 2, 3, 4:
\t1) repeat
\t2) -
\t3) -
\t4) exit
"""
    action = input(prompt)
    while action not in {'1','2','3','4'}:
        action = input("Wrong input, enter again: ")

    return int(action)
