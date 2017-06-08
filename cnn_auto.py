from PIL import Image
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils.np_utils import to_categorical
import numpy as np
import h5py
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

img_width=28
img_height=28

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
x_train=np.reshape(train_data,(train_num,img_width,img_height,1))
# read test data to 2-D array -> val_data , val_label
for filename2 in os.listdir(dir_val):
    tmp_path2=dir_val+'/'+filename2
    for tmp2 in os.listdir(tmp_path2):
        allpath2=tmp_path2+'/'+tmp2
        im3=Image.open(allpath2)
        im4=im.resize((img_width,img_height))
        val_data.append(np.array(im2))
        val_label.append(label2)
    label2=label2+1
x_test=np.reshape(val_data,(validation_num,img_width,img_height,1))
#divide into ten classes
train_label=to_categorical(train_label,num_class)



input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=25,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HD
autoencoder.save_weights("model.h5")
print("Saved model to disk")

