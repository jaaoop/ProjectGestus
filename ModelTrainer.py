import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
from os.path import join, split
import glob

# Argument parser for easy modifications
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--chart',
                    required=False, default=False,
                    help="Keep train chart saved")
arguments = vars(parser.parse_args())

# Show train loss and accuracy
class Callback(tf.keras.callbacks.Callback):
    first_pass = True
    def on_epoch_end(self, epoch, logs={}):
        plt.scatter(epoch, logs.get('loss'), color='blue', marker='.', label = 'Loss')
        plt.scatter(epoch, logs.get('acc'), color='red', marker='.', label = 'Accuracy')
        if self.first_pass:
            plt.legend()
            self.first_pass = False

        plt.savefig('temp_chart.png')
        img = cv2.imread('temp_chart.png')
        cv2.imshow('Logs', img)
        cv2.waitKey(1)

testFolder = []
trainFolder = []

# If train folder or/and test folder does not exist, exit
if not os.path.exists(join('Dataset','Train')) or not os.path.exists(join('Dataset','Test')):
    print('[ERROR] Train or/and Test path does not exist')
    exit()

# Get name of train and test folders
for folder in glob.glob(join(join('Dataset', 'Train'), '*')):
    trainFolder.append(split(folder)[-1])
    
for folder in glob.glob(join(join('Dataset', 'Test'), '*')):
    testFolder.append(split(folder)[-1])

# If the number of train folders is different than test folders, exit
testFolder.sort()
trainFolder.sort()

if not trainFolder == testFolder:
    print("[ERROR] Train and test folders are different")
    exit()
else:
    numGestures = len(trainFolder)

# Get images for training and testing
testImages_count = 0
trainImages_count = 0
for _, folder, images in os.walk(join('Dataset','Train')):
    trainImages_count+=len(images)

for _, folder, images in os.walk(join('Dataset','Test')):
    testImages_count+=len(images)

batch_size = 16

# Data augmentation for train images
trainImagesGen = ImageDataGenerator(rescale=1.0/255.0, 
                                    rotation_range=20, 
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    fill_mode='nearest'
                                  )

# Load train images
trainImages = trainImagesGen.flow_from_directory(join('Dataset','Train'), 
                                                    batch_size=batch_size, 
                                                    target_size=(89,100), 
                                                    class_mode='categorical', 
                                                    color_mode = 'grayscale')

# Data augmentation for test images, only use rescale
testImagesGen = ImageDataGenerator(rescale=1.0/255.0)

# Load test images 
testImages = testImagesGen.flow_from_directory(join('Dataset','Test'), 
                                                batch_size=batch_size, 
                                                target_size=(89,100), 
                                                class_mode='categorical', 
                                                color_mode = 'grayscale')

# Sequence of convolutions for the model
model = Sequential([Conv2D(32, (2,2), activation = 'relu', input_shape = [89, 100, 1]), 
                    MaxPooling2D(2,2), 

                    Conv2D(64, (2,2), activation='relu'), 
                    MaxPooling2D(2,2), 

                    Conv2D(128, (2,2), activation='relu'), 
                    MaxPooling2D(2,2), 

                    Conv2D(256, (2,2), activation='relu'), 
                    MaxPooling2D(2,2), 

                    Conv2D(256, (2,2), activation='relu'), 
                    MaxPooling2D(2,2), 

                    Flatten(),
                    Dense(1000, activation='relu'),
                    Dropout(.75),
                    Dense(numGestures, activation='softmax')])

# Define the CNN model
adam = Adam(learning_rate = 0.001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['acc'])

callback = Callback()

# Start training
history = model.fit_generator(trainImages, 
                                steps_per_epoch = trainImages_count/batch_size, 
                                epochs=50,  
                                validation_data=testImages,
                                validation_steps=testImages_count/batch_size,
                                verbose = 1,
                                callbacks = [callback])

# Save model weights
model.save(join("ModelWeights","GestureRecogModel_tf.tfl"))

# Remove temp_chart image
if arguments['chart'] == False:
    os.remove('temp_chart.png')