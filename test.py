import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

class Callback(tf.keras.callbacks.Callback):
    first_pass = True
    def on_epoch_end(self, epoch, logs={}):
        plt.scatter(epoch, logs.get('loss'), color='blue', marker='.', label = 'Loss')
        plt.scatter(epoch, logs.get('acc'), color='red', marker='.', label = 'Class')
        if self.first_pass:
            plt.legend()
            self.first_pass = False

        plt.savefig('temp.png')
        img = cv2.imread('temp.png')
        cv2.imshow('Logs', img)
        cv2.waitKey(1)

testImages_count = 0
trainImages_count = 0

for _, _, images in os.walk('Dataset/Train'):
    trainImages_count+=len(images)

for _, _, images in os.walk('Dataset/Test'):
    testImages_count+=len(images)

batch_size = 16

trainImagesGen = ImageDataGenerator(rescale=1.0/255.0, 
                                    rotation_range=20, 
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    fill_mode='nearest'
                                  )
trainImages = trainImagesGen.flow_from_directory('Dataset/Train', 
                                                    batch_size=batch_size, 
                                                    target_size=(89,100), 
                                                    class_mode='categorical', 
                                                    color_mode = 'grayscale')

testImagesGen = ImageDataGenerator(rescale=1.0/255.0)

testImages = testImagesGen.flow_from_directory('Dataset/Test', 
                                                batch_size=batch_size, 
                                                target_size=(89,100), 
                                                class_mode='categorical', 
                                                color_mode = 'grayscale')

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

                    # Conv2D(256, (1,1), activation='relu'), 
                    # MaxPooling2D(2,2),

                    # Conv2D(128, (2,2), activation='relu'), 
                    # MaxPooling2D(2,2), 

                    # Conv2D(64, (2,2), activation='relu'),
                    # MaxPooling2D(2,2),

                    Flatten(),
                    Dense(1000, activation='relu'),
                    Dropout(.75),
                    Dense(4, activation='softmax')])

# model.summary()

# Define the CNN Model
adam = Adam(learning_rate = 0.001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['acc'])

callback = Callback()
history = model.fit_generator(trainImages, 
                                steps_per_epoch = trainImages_count/batch_size, 
                                epochs=50,  
                                validation_data=testImages,
                                validation_steps=testImages_count/batch_size,
                                verbose = 1,
                                callbacks = [callback])

model.save("TrainedModel/GestureRecogModel_tf.tfl")