#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Functions import *
from ReadImages import *


# In[2]:


class ThreeLayerNet_1:
    '''
    ReLU - ReLU - Sigmoid
    
    '''
    ##### パラメータのセット #####
    def __init__(self, input_size, hidden_size=25, output_size=1, sigma=0.01):
        self.params = {}
        self.params['W1'] = sigma * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = sigma * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = sigma * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
    
    ##### 順伝播 #####
    def _forward(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
    
        a1 = np.dot(x, W1) + b1
        z1 = ReLU(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = ReLU(a2)
        a3 = np.dot(z2, W3) + b3
        y = sigmoid(a3)
        return y
    
    ##### 損失 #####
    def _loss(self, x, t):
        y = self._forward(x)
        return cross_entropy_error(y.reshape(-1, ), t)
    
    ##### 認識精度 #####
    def _accuracy(self, x, t):
        y = self._forward(x)
        acc = 1 - abs(y-t)
        return np.mean(acc)
    
    ##### 逆伝播 #####
    def _gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        batch_num = x.shape[0]
        #print(batch_num) #
        
        # forward
        a1 = np.dot(x, W1) + b1
        #print('a1:', np.shape(a1)) #
        z1 = ReLU(a1)
        #print('z1:', np.shape(z1)) #
        a2 = np.dot(z1, W2) + b2
        #print('a2:', np.shape(a2)) #
        z2 = ReLU(a2)
        #print('z2:', np.shape(z2)) #
        a3 = np.dot(z2, W3) + b3
        #print('a3:', np.shape(a3)) #
        y = sigmoid(a3)
        #print(' y:', np.shape(y)) # (3, 1)
        
        # backward
        grads = {}
        #print(' t:', np.shape(t))
        dy = (y - t.reshape(batch_num, 1)) / batch_num
        #print('dy:', np.shape(dy)) # (3, 10)
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        
        dz2 = np.dot(dy, W3.T)
        da2 = ReLU_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        
        dz1 = np.dot(dz2, W2.T)
        da1 = ReLU_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
        


# In[2]:


class ThreeLayerNet_2:
    '''
    ReLU - ReLU - softmax
    
    '''
    ##### パラメータのセット #####
    def __init__(self, input_size, hidden_size=25, output_size=1, sigma=0.01):
        self.params = {}
        self.params['W1'] = sigma * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = sigma * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = sigma * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
    
    ##### 順伝播 #####
    def _forward(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
    
        a1 = np.dot(x, W1) + b1
        z1 = ReLU(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = ReLU(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y
    
    ##### 損失 #####
    def _loss(self, x, t):
        y = self._forward(x)
        return cross_entropy_error(y.reshape(-1, ), t)
    
    ##### 認識精度 #####
    def _accuracy(self, x, t):
        y = self._forward(x)
        acc = 1 - abs(y-t)
        return np.mean(acc)
    
    ##### 逆伝播 #####
    def _gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        batch_num = x.shape[0]
        #print(batch_num) #
        
        # forward
        a1 = np.dot(x, W1) + b1
        #print('a1:', np.shape(a1)) #
        z1 = ReLU(a1)
        #print('z1:', np.shape(z1)) #
        a2 = np.dot(z1, W2) + b2
        #print('a2:', np.shape(a2)) #
        z2 = ReLU(a2)
        #print('z2:', np.shape(z2)) #
        a3 = np.dot(z2, W3) + b3
        #print('a3:', np.shape(a3)) #
        y = softmax(a3)
        #print(' y:', np.shape(y)) # (3, 1)
        
        # backward
        grads = {}
        #print(' t:', np.shape(t))
        dy = (y - t.reshape(batch_num, 1)) / batch_num
        #print('dy:', np.shape(dy)) # (3, 10)
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        
        dz2 = np.dot(dy, W3.T)
        da2 = ReLU_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        
        dz1 = np.dot(dz2, W2.T)
        da1 = ReLU_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
        


# In[ ]:





# In[2]:


class ThreeLayerNet_3:
    '''
    Sigmoid - Sigmoid - Sigmoid
    
    '''
    ##### パラメータのセット #####
    def __init__(self, input_size, hidden_size=25, output_size=1, sigma=0.01):
        self.params = {}
        self.params['W1'] = sigma * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = sigma * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = sigma * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
    
    ##### 順伝播 #####
    def _forward(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = sigmoid(a3)
        return y
    
    ##### 損失 #####
    def _loss(self, x, t):
        y = self._forward(x)
        return cross_entropy_error(y.reshape(-1, ), t)
    
    ##### 認識精度 #####
    def _accuracy(self, x, t):
        y = self._forward(x)
        acc = 1 - abs(y-t)
        return np.mean(acc)
    
    ##### 逆伝播 #####
    def _gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        batch_num = x.shape[0]
        #print(batch_num) #
        
        # forward
        a1 = np.dot(x, W1) + b1
        #print('a1:', np.shape(a1)) #
        z1 = sigmoid(a1)
        #print('z1:', np.shape(z1)) #
        a2 = np.dot(z1, W2) + b2
        #print('a2:', np.shape(a2)) #
        z2 = sigmoid(a2)
        #print('z2:', np.shape(z2)) #
        a3 = np.dot(z2, W3) + b3
        #print('a3:', np.shape(a3)) #
        y = sigmoid(a3)
        #print(' y:', np.shape(y)) # (3, 1)
        
        # backward
        grads = {}
        #print(' t:', np.shape(t))
        dy = (y - t.reshape(batch_num, 1)) / batch_num
        #print('dy:', np.shape(dy)) # (3, 10)
        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)
        
        dz2 = np.dot(dy, W3.T)
        da2 = sigmoid_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)
        
        dz1 = np.dot(dz2, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




