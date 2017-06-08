from PIL import Image
from scipy import misc
#from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os

dir_train="/data2/chenwu/Sketch-recognition-using-deep-learning/tendata/train"

dir_val="/data2/chenwu/Sketch-recognition-using-deep-learning/tendata/validation"

train_data=[]
train_label=[]
val_data=[]
val_label=[]

train_num=700
validation_num=100
num_class=10

img_width=128
img_height=128

label=0
label2=0
cnt=0
cnt2=0

# read train data to 2-D array -> train_data , train_label
for filename in os.listdir(dir_train):
    tmp_path=dir_train+'/'+filename
    #tmp_train_data=[]
    #tmp_train_label=[]
    for tmp in os.listdir(tmp_path):
        allpath=tmp_path+'/'+tmp
        im=Image.open(allpath)
        im2=im.resize((img_width,img_height))
        train_data.append(np.array(im2))
        train_label.append(label)
    #train_data.append(tmp_train_data);
    #train_label.append(tmp_train_label);

    label=label+1
fin_train_data=np.reshape(train_data,(train_num,img_width,img_height,1))
# read test data to 2-D array -> val_data , val_label
for filename2 in os.listdir(dir_val):
    tmp_path2=dir_val+'/'+filename2
    #tmp_val_data=[]
    #tmp_val_label=[]
    for tmp2 in os.listdir(tmp_path2):
        allpath2=tmp_path2+'/'+tmp2
        im3=Image.open(allpath2)
        im4=im.resize((img_width,img_height))
        val_data.append(np.array(im2))
        val_label.append(label2)
    #val_data.append(tmp_val_data)
    #val_label.append(tmp_val_label)
    label2=label2+1
fin_val_data=np.reshape(val_data,(validation_num,img_width*img_height*1))
#divide into ten classes
train_label=to_categorical(train_label,num_class)

#print  fin_train_data.ndim
#print  fin_train_data.shape

model = Sequential()
model.add(Convolution2D(16,11, 11, input_shape=( img_width, img_height,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32,5,5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(num_class))
model.add(Activation('softmax'))

sgd=optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history=model.fit(fin_train_data,train_label, batch_size=50)
#loss_and_metrics = model.evaluate(val_data,val_label, batch_size=10)
#classes = model.predict(fin_val_data, batch_size=10)
