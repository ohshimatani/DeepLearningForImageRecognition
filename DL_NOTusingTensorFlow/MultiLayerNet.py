#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import OrderedDict
from ReadImages import *
from Layers import *


# In[4]:


class MultiLayerNet:
    def __init__(self, input_size, hidden_layer_num, hidden_size, output_size, activation='ReLU'):
        self.input_size = input_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        act = {'ReLU':ReLU, 'Sigmoid':Sigmoid}
        self.activation = act[activation]
        self.params = {}
        
        self.__init_weight(self.hidden_layer_num)
        self.layers = OrderedDict
        for i in range(1, self.hidden_size):
            self.layers['Affine_' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            self.layers[str(self.activation) + '_' + str(i)] = activation()
        
        self.last_layer = SoftmaxWithLoss()
        #self.loss_layer = Loss(classification=True, regression=False)
    
    def __init_weight(self, n):
        if self.activation == ReLU:
            sigma = sqrt(2/n)
        elif self.activation == Sigmoid:
            sigma = sqrt(1/n)
            
        self.params['W1'] = self.input_size * np.random.randn(self.input_size, n)
        self.params['b1'] = np.zeros(1)
        for i in range(2, self.hidden_size):
            self.params['W' + str(i)] = sigma * np.random.randn(n, n)
            self.params['b' + str(i)] = np.zeros(n)
    
    def predict(self, x):
        for layer in self.layers.value():
            x = layer.forward(x)
            
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self, x, t):
        y = self.loss(x, t)
        
        L = 1
        L = self.last_layer.backward(L)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            L = layer.backward(L)
        
        grads = {}
        for idx in range(1, self.hidden_layer_size+2):
            grads['W' + str(idx)] = self.layers['Affine_' + str(idx)].dW * self.layers['Affine_' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine_' + str(idx)].db

        return grads


# In[ ]:




