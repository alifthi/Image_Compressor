'''
Title: image compression
Author: alifthi    
Date Created: 2021-01-11
Describe: Impelement an image compressor with an Auto Encoder model
'''

import keras as ks
from keras.layers.convolutional import Conv2D as cv
from keras.layers.convolutional import Conv2DTranspose as cvt
from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt
from  keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

(x,_),(xt,_) = mnist.load_data()
clbk = [TensorBoard(log_dir='./logs')]
 
model = ks.models.Sequential()
model.add(cv(16,kernel_size=3,strides=2,padding='same'))
model.add(cv(1,kernel_size=3,strides=2,padding='same'))
encoder = model

model = ks.models.Sequential()
model.add(cvt(8,kernel_size=3,strides=2,padding='same'))
model.add(cvt(16,kernel_size=3,strides=2,padding='same'))

decoder = model
encoder.compile(optimizer='adam', loss='binary_crossentropy',)
decoder.compile(optimizer='adam', loss='binary_crossentropy')

model = ks.models.Sequential()
model.add(encoder)
model.add(decoder)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
x = np.expand_dims(x,axis=3) 
x = x/255.0
xt = np.expand_dims(xt,axis=3) 
xt = xt/255.0
model.fit(x,x,epochs=5,batch_size=128,callbacks=clbk)

plt.imshow(xt[0]) 
cmpIm = encoder.predict(xt)
plt.imshow(xt[0]) 
plt.imshow(cmpIm[0,:,:,0],cmap='gray')
model.save('Compression.h5')
