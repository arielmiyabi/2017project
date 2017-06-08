from PIL import Image
from scipy import misc
#from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os

train_data=[]
train_label=[]
valid_data=[]
valid_label=[]
label=0
img_width=28
img_height=28
num_class=10
cnt=0
dirpath="/home/chenwu2/ten"
for filename in os.listdir(dirpath):
    path=dirpath+'/'+filename
    temp_load_file=np.load(path)
    load_file=temp_load_file[1:5001]
    print load_file.shape
    if (cnt>0):
        train_data=np.concatenate((train_data, load_file))
    else:
        train_data=load_file
    #print train_data.shape
    for i in range(1,5000+1):
        train_label.append(label)
        cnt=cnt+1
    #print cnt
    label=label+1
    if (label>=10):
        break

fin_train_data=np.reshape(train_data,(cnt,img_width,img_height,1))
#fin_train_data=np.reshape(train_data,(train_num,img_width,img_height,1))

print fin_train_data.shape

train_label=to_categorical(train_label,num_class)
model = Sequential()

model.add(Convolution2D(64,3, 3, input_shape=(img_width, img_height,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64,1,1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))


model.add(Flatten())

model.add(Dense(75))
model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(784))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(num_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(fin_train_data,train_label, batch_size=100,validation_split=0.2)


#loss_and_metrics = model.evaluate(val_data,val_label, batch_size=10)
#classes = model.predict(fin_val_data, batch_size=10)
