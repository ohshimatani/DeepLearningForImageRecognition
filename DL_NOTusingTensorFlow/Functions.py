#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def ReLU(x):
    return np.maximum(x, 0)


# In[1]:


def ReLU_grad(x):
    grad = (x>=0).astype(np.float)
    return grad


# In[4]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  


# In[ ]:


def sigmoid_grad(x):
    return (1 - sigmoid(x)) * sigmoid(x)


# In[5]:


def cross_entropy_error(y, t):
    L = -t*np.log(y) - (1-t)*np.log(1-y)
    return np.sum(L) / L.shape[0]


# In[ ]:


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# In[ ]:




