import numpy as np
import sys

# read user input
def readArgs():
    if len(sys.argv) == 1:
        fileName = input('Enter dataset path: ')
    elif len(sys.argv) == 3 and sys.argv[1] == '-d':
        fileName = sys.argv[2]
    else:
        raise Exception('Error, check command line arguments')

    # hyperparameters
    # filtersNum = int(input('Enter number of filters: '))
    # filterSize = int(input('Enter size of each filter: '))
    convNum = int(input('Enter number of convolution layers: '))
    epochs = int(input('Enter number of epochs: '))
    batchSize = int(input('Enter batch size: '))

    return fileName, convNum, epochs, batchSize


# 4 bytes per integer in data files, 1 byte per pixel and per label
bytesInt = 4

#Method that reads images from input file
def readImages(fileName):
    with open(fileName, mode='rb') as f:  # open the file in binary mode
        f.seek(bytesInt)  # skip magic number
        imgNum = int.from_bytes(f.read(bytesInt), byteorder='big') # number of images
        rows = int.from_bytes(f.read(bytesInt), byteorder='big') # number of rows
        cols = int.from_bytes(f.read(bytesInt), byteorder='big') # number of columns

        print("Images in file: {}\nDimensions: {}x{}\n".format(imgNum, rows, cols), end='')

        # read all pixels at once
        images = np.fromfile(f,np.uint8)
        # 4-d array of images, 3-d array of size (28 x 28 x 1) per gray-scale image , i.e. one channel
        images = np.reshape(images.astype('float32'), (imgNum, rows, cols, 1))

        return images, imgNum, rows, cols

#Method that reads labels from input file
def readLabels(fileName):
    with open(fileName, mode='rb') as f:  # open the file in binary mode
        f.seek(bytesInt)  # skip magic number
        lblNum = int.from_bytes(f.read(bytesInt), byteorder='big') # number of labels

        # read all labels at once
        labels = np.fromfile(f,np.uint8)

        return labels, lblNum

