import os
from PIL import Image
from cProfile import label
import math
from pickletools import optimize
from typing import Counter
from unittest.mock import patch
import matplotlib.image as img
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 
import imutils

train_dir = f'IMAGENES_IA_CUBIERTOS\_train'
classes = os.listdir(train_dir)

x = []
y = []
listaRotaciones = [0,90,180,270]
contador = 0
for class_name in classes: 
    class_samples = os.listdir(f'{train_dir}/{class_name}')
    for sample in class_samples:
        
      
        
        image = cv2.resize(cv2.imread(f'{train_dir}/{class_name}/{sample}'),(200,200))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for grados in listaRotaciones:
            rows,cols,colors = image.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),grados,1)
            image = cv2.warpAffine(image,M,(cols,rows))
            imagen_normalizar = tf.cast(image,tf.float32)
            imagen_normalizar /= 255
            # print(contador)
            x.append(imagen_normalizar)
            y.append(contador)
    print(class_name)
    
    contador = contador +1 

x = np.asarray(x)
y= np.array(y,dtype=int)
# for i in range(0,20):
#     cv2.imshow('a',x[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# plt.figure()
# plt.imshow(x[30], cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

print('Ejecutando modelo convolucional')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(50,(3,3),input_shape= (200,200,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Conv2D(100,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Conv2D(100,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Conv2D(100,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(3,3),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(100,activation = tf.nn.relu),
    
    tf.keras.layers.Dense(3,activation = tf.nn.softmax),
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# x = x.repeat().batch(lotes)
lotes  = 8
historial = model.fit(x,y, epochs=40, steps_per_epoch= math.ceil(len(x)/lotes))

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history['loss'])
plt.show()

model.save('./modelo/modelo.h5')
model.save_weights('./modelo/pesos.h5')

