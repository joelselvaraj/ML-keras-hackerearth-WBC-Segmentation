from __future__ import print_function

import os
import numpy as np

import cv2

data_path = 'Data\\'

image_rows = 128
image_cols = 128


def create_train_data():
    train_data_path = os.path.join(data_path, 'Train_Data')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '-mask.jpg'
        
        img = cv2.imread(os.path.join(train_data_path, image_name))
        img = cv2.resize(img,(image_rows,image_cols))
        
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask,(image_rows,image_cols))
        
        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print(imgs_mask.shape)
    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    test_data_path = os.path.join(data_path, 'Test_Data')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 3), dtype=np.uint8)
    imgs_id = np.ndarray((total), dtype=np.object)
    imgs_size = np.ndarray((total), dtype=np.object)
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img = cv2.imread(os.path.join(test_data_path, image_name))
        img_size = str(img.shape[0]) +","+str(img.shape[1])
        img = cv2.resize(img,(image_rows,image_cols))
        img = np.array([img])
        img_id = image_name.split('.')[0] + "-mask"
        
        imgs_id[i] = img_id
        imgs[i] = img
        imgs_size[i] = img_size
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print(imgs_id)
    print(imgs_size)
    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    np.save('imgs_size.npy',imgs_size)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    imgs_size = np.load('imgs_size.npy')
    return imgs_test,imgs_id,imgs_size

if __name__ == '__main__':
    create_train_data()
    create_test_data()
