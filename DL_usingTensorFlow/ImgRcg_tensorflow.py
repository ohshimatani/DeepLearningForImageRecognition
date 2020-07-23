#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from ReadImages import *


# In[2]:


convertPNG2JPG(path_list)
(x_train, t_train), (x_test, t_test) = read_image()


# In[3]:


if __name__ == '__main__':
    print(np.shape(x_train), np.shape(t_train), np.sum(t_train))
    print(np.shape(x_test), np.shape(t_test), np.sum(t_test))


# In[4]:


model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(256, 256, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])


# In[5]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[6]:


model.fit(x_train, t_train, epochs=10)


# In[7]:


test_loss, test_acc = model.evaluate(x_test, t_test)
print('Test accuracy:', test_acc)


# In[ ]:




