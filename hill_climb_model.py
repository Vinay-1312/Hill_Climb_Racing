# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:53:42 2020

@author: DELL
"""


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Conv2D,MaxPool2D,Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_acc') > 0.95):   
            print("\nReached accuracy, so stopping training!!")   
            self.model.stop_training = True

train_datagen=ImageDataGenerator(rescale=1.0/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
callback=myCallback()#stop training when 0.95 accuracy is reached
test_datagen=ImageDataGenerator(rescale=1.0/255)
#Data Loading

train_data=train_datagen.flow_from_directory("rps",target_size=(150,150),batch_size=32,class_mode="binary")
test_data=test_datagen.flow_from_directory("rps-test-set",target_size=(150,150),batch_size=32,class_mode="binary")

cnn=Sequential()
cnn.add(Conv2D(filters=74,kernel_size=(3,3),input_shape=(150,150,3),activation="relu"))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(filters=74,kernel_size=(3,3),activation="relu"))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))
cnn.add(Dropout(0.2))

cnn.add(Conv2D(filters=74,kernel_size=(3,3),activation="relu"))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))
cnn.add(Dropout(0.2))

cnn.add(Flatten())
cnn.add(Dense(128,activation="relu"))
cnn.add(Dense(2,activation="softmax"))
cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["acc"])
cnn.fit_generator(train_data,validation_data=test_data,epochs=10,callbacks=[callback])
cnn.save('my_model2')