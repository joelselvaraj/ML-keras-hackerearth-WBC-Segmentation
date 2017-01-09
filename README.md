# ML-keras-hackerearth-WBC-Segmentaion
Code for WBC Segementation, a hackerearth compeition held by sigtuple
## Requirements:
 - python 3.5
 - anaconda 4.2.0 
 - tensorflow - conda 
 - keras
 - theano

## To Run:
 - Put the train data and test data folder in the Data folder
 - First run the data.py file to load the data and preprocess it.
 - Then run the train.py file to train a Convolutional Neural Network, make prediction and store the predicted images.
 - The predicted mask images can be found in the Data/output folder.

## Approach:
 - The images are converted into arrays and stored.
 - A Convolutional Neural Network(CNN) having 5 convolution2d block of filter values 64, 32 and 1. Each layer is seperated by an activation layer(Relu).
 - The CNN is trained with the training data and then made to predict the output(mask images) for the testing data.
 - The output is saved back as an image in the Data/output folder.
 
