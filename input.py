import numpy as np

# 4 bytes per integer in data files, 1 byte per pixel and per label
bytesInt = 4

#Function that reads images from input file
def readImages(fileName):
    with open(fileName, mode='rb') as f:  # open the file in binary mode
        f.seek(bytesInt)  # skip magic number
        imgNum = int.from_bytes(f.read(bytesInt), byteorder='big') # number of images
        rows = int.from_bytes(f.read(bytesInt), byteorder='big') # number of rows
        cols = int.from_bytes(f.read(bytesInt), byteorder='big') # number of columns

        print("Images in file: {}\nDimensions: {}x{}\n".format(imgNum, rows, cols), end='')

        # read all pixels at once
        images = np.fromfile(f,np.uint8)
        # 3-d array of images, 2-d array of size (rows x cols) per image
        images = images.reshape([imgNum, rows, cols])

        return images, imgNum, rows, cols

#Function that reads labels from input file
def readLabels(fileName):
    with open(fileName, mode='rb') as f:  # open the file in binary mode
        f.seek(bytesInt)  # skip magic number
        lblNum = int.from_bytes(f.read(bytesInt), byteorder='big') # number of labels

        # read all labels at once
        labels = np.fromfile(f,np.uint8)

        return labels, lblNum
