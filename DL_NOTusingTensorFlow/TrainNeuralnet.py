#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from ReadImages import *
from Layers import *
from MultiLayerNet import *


# In[5]:


(x_train, t_train), (x_test, t_test) = read_image()


# In[7]:


network = MultiLayerNet(input_size=(256, 256), hidden_layer_num=15, hidden_size=5, output_size=2, activation='ReLU')


# In[ ]:




