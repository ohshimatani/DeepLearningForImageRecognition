#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


'''class :
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
''';


# In[3]:


class Relu:
    def __init__(self):
        pass
    
    def forward(self, x):
        return np.max(x, 0)
    
    def backward(self, L):
        return np.max(L, 0)


# In[4]:


class Sigmoid:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backward(self, L):
        return L * self.y * (1-self.y)


# In[5]:


class Tanh:
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        self.x = x
        return np.tanh(x)
    
    def backward(self, L):
        return L * 1 / np.cosh(self.x)**2


# In[6]:


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, L):
        out = np.dot(L, self.W.T)
        self.dW = np.dot(self.x.T, L)
        self.db = np.sum(L, axis=0)
        return out


# In[1]:


class Softmax:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        c = np.max(x)
        exp_x = np.exp(x - c)
        sum_exp_x = np.sum(exp_x)
        self.y = exp_x / sum_exp_x
        return self.y
    
    def backward(self, L):
        pass


# In[7]:


class Loss:
    '''
    Classification: cross_entropy_error
        Regression: square_root_error
       
    '''
    def __init__(self, Classification=True, Regression=False):
        pass
        
    def forward(self, y, t):
        if Regression:
            # mean_squared_error
            loss = 1/2 * np.sum((y - t)**2)
            return loss
        else:
            # cross_entropy_error
            loss = -np.sum(t * np.log(y))
            return loss


# In[ ]:


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, L=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


# In[ ]:





# In[ ]:





# In[8]:


class Dropout:
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass


# In[9]:


class BatchNormalization:
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass


# In[10]:


class Convolusion:
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass


# In[11]:


class Pooling:
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass


# In[ ]:





# In[ ]:





# In[ ]:




