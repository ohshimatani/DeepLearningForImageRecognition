#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from ThreeLayerNet import *


# In[2]:


convertPNG2JPG(path_list)
(x_train, t_train), (x_test, t_test) = read_image()


# In[3]:


if __name__ == '__main__':
    print(np.shape(x_train), np.shape(t_train), np.shape(x_test), np.shape(t_test))
    print(len(x_train))


# In[4]:


train_suffle_index = np.random.permutation(len(x_train))
test_suffle_index = np.random.permutation(len(x_test))
(x_train, t_train), (x_test, t_test) = (x_train[train_suffle_index], t_train[train_suffle_index]), (x_test[test_suffle_index], t_test[test_suffle_index])


# In[5]:


# ハイパーパラメータ
iterations = 300
train_size = x_train.shape[0]
batch_size = 5
learning_late = 0.01


# In[6]:


if __name__ == '__main__':
    
    # カラーかグレースケールかの設定
    color = True
    
    if color:
        inputsize = 256*256*3
    else:
        inputsize = 256*256


# In[7]:


'''
ThreeLayerNet_1 : ReLU-ReLU-Sigmoid
ThreeLayerNet_2 : ReLU-ReLU-Softmax
ThreeLayerNet_3 : Sigmoid-Sigmoid-Sigmoid

'''

network = ThreeLayerNet_1(input_size=inputsize, hidden_size=100, output_size=1, sigma=0.01)


# In[8]:


train_loss_list = []
train_acc_list = []
test_acc_list = []


# In[9]:


for i in range(iterations):
    ##### 学習 #####
    batch = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch]
    t_batch = t_train[batch]
    
    grad = network._gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        # SGD
        #print(np.shape(network.params[key]))
        network.params[key] -= learning_late * grad[key]
        
    loss = network._loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    ##### 認識精度 #####
    if i % 10 == 0:
        #loss = network._loss(x_batch, t_batch)
        #train_loss_list.append(loss)
        train_acc = network._accuracy(x_train, t_train)
        test_acc = network._accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print([i], train_acc,',', test_acc, ',', loss)
        
print('---------- Done ----------')


# In[10]:


if __name__ == '__main__':
    
    plt.figure(figsize=(15, 5),dpi=75)
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.xlabel('epocs')
    plt.ylabel('loss')
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(train_acc_list)), train_acc_list, color='red', label='train_acc')
    plt.plot(np.arange(len(test_acc_list)), test_acc_list, color='blue', label='test_acc')
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel('epocs')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    
    plt.show()


# In[ ]:





# In[11]:


##### テストデータでの評価 #####


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




