from __future__ import print_function

import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, Reshape, UpSampling2D, merge)
from keras.models import Sequential
from keras.optimizers import SGD

from data import load_test_data, load_train_data

K.set_image_dim_ordering('tf')  # Theano dimension ordering in this code

img_rows = 128
img_cols = 128

smooth = 1.

def get_model(X_train):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(128,128,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(128,128,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(1, 1, 1, border_mode='same'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    return model


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    print(imgs_mask_train.shape)
    imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], img_rows, img_cols, 1)
    imgs_mask_train = imgs_mask_train.reshape(imgs_mask_train.shape[0], img_rows, img_cols, 1)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    print(imgs_mask_train.shape)
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_model(imgs_train)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id, imgs_size = load_test_data()

    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization
    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print(imgs_test.shape)
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    imgs_mask_test *= 255
    i=0
    for img,name,size in zip(imgs_mask_test,imgs_id,imgs_size):
        img=cv2.resize(img, (int(size.split(',')[1]) , int(size.split(',')[0])))
        ret,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        cv2.imwrite("Data/output/"+str(name) +".jpg", img )
        i+=1
    print(imgs_mask_test.shape)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
