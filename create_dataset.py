import os
import numpy as np
import pandas as pd

from PIL import Image
from keras.preprocessing.image import img_to_array,load_img

from progress.bar import IncrementalBar


def create_dataset(dir,height,width,channels):
    # contains list of all image files
    file_to_train = []
    # contains corrosponding labels ot the image files
    y_train = []

    # creating the list of all files(file_to_train) and their corrosponding labels(y_train)
    bar = IncrementalBar('Reading Files',max=len(os.listdir(dir)))
    for image in os.listdir(dir):
        file_to_train.append(image)
        label = image.split('.')[0]
        y_train.append(label)
        bar.next()
    bar.finish()

    print('TOTAL NUMBER OF FILES TO BE TRAINED : ',len(file_to_train))

    # initializing the train data with a zero array
    X_train = None
    X_train = np.ndarray(shape = (len(file_to_train),height,width,channels), dtype = np.float32)

    print('SHAPE OF TRAINING DATA : ',X_train.shape)

    # initializing the progress bar
    bar = IncrementalBar('Creating Data',max=len(file_to_train))

    # creating the train dataset(np array of all images in file_to_train)
    i=0
    for image in file_to_train:
        img = load_img(dir + '/' + image)
        img = img.resize((width, height), Image.ANTIALIAS)

        x = img_to_array(img)
        x = x.reshape(height,width,channels)

        X_train[i] = x
        i+=1
        bar.next()

    bar.finish()

    # dividing the data into train and validation data
    # one could also use sklearn.model_selection.train_test_split for this
    (X_train,X_val) = X_train[:21999],X_train[22000:]
    (y_train,y_val) = y_train[:21999],y_train[22000:]

    return (X_train,X_val,y_train,y_val)
