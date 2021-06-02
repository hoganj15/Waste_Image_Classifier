# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:09:18 2020

@author: arguz
"""

#pip install tensorflow
#pip install Keras

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#image preprocessing -- Training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#image preprocessing -- Test
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

valid_set = train_datagen.flow_from_directory('data/validation', 
                                              target_size=(64, 64), 
                                              batch_size=32, 
                                              class_mode='binary', 
                                              subset='validation')

############ CNN ############

cnn = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[64, 64, 3]),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.4),
    
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.4),
    
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Dropout(0.4),
    
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.6),
  tf.keras.layers.Dense(6, activation='softmax')
])

#compiling the CNN
cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training CNN and evaluation on test set
cnn.fit(x = training_set, validation_data = valid_set, epochs = 3)

############ Testing ############

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/Users/arnaudguzman-annes/Desktop/Spring 2021/MGSC677 - Intro to AI and Deep Learning II/Final Project/test1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices

#Result of the CNN
#[acces the batch][access the prediction]
if result[0][0] == 1:
  prediction = 'trash'
else:
  prediction = 'recycle'
   
print(prediction)