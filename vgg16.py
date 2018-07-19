#import packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

X_train = None                  #train data of shape (no. of examples, shape of each image)
Y_train = None                  #One-hot representation of target matrix
X_test = None
Y_test = None

input_shape = None              #shape of input image
num_epochs = None               #no. of epochs
num_batch_size = None           #batch size, 32/64/128 are prefered


#create VGG16 model:
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), 2))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), 2))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), 2))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))

model.add(MaxPooling2D((2,2), 2))

model.add(Flatten())

model.add(Dense(units=4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='softmax'))                        #change 1000 with no. of classes in given example


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=num_batch_size)
model.evaluate(X_test, Y_test)
