#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''1_ReadImage
#画像データをnumpy配列にして読み込む#

_convert_numpy:
    指定された画像(xxx.jpg)をnumpy配列に変換

_make_dataset:
    label0とlabel1を結合(0, 1の順に)

_make_labeldata:
    ラベルを作る

''';


# In[2]:


# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import glob
import matplotlib.pyplot as plt
#import cv2
from PIL import Image
from ConvertPNG2JPG import *


# In[3]:


#それぞれのファイルのパスを所得
train_path_0 = os.getcwd() + '\\' + '_data' + '\\' + 'train_images' + '\\' + 'not_kumamon'
train_path_1 = os.getcwd() + '\\' + '_data' + '\\' + 'train_images' + '\\' + 'kumamon'
test_path_0 = os.getcwd() + '\\' + '_data' + '\\' + 'test_images' + '\\' + 'not_kumamon'
test_path_1 = os.getcwd() + '\\' + '_data' + '\\' + 'test_images' + '\\' + 'kumamon'

path_list = [train_path_0, train_path_1, test_path_0, test_path_1]


# In[4]:


def _convert_numpy(file_path, flat, color):
    if color:
        files = glob.glob(file_path + './*.jpg')
        array = np.array([])
        i = 0
        for file in files:
            if flat:
                img = np.array(Image.open(file)).flatten()
            else:
                img = np.array(Image.open(file))

            if i == 0:
                array = np.array([img])
            else:
                array = np.append(array, [img], axis=0)
            i += 1
        return array
    
    else:
        files = glob.glob(file_path + './*.jpg')
        array = np.array([])
        i = 0
        for file in files:
            if flat:
                img = np.array(Image.open(file).convert('L')).flatten()
            else:
                img = np.array(Image.open(file).convert('L'))

            if i == 0:
                array = np.array([img])
            else:
                array = np.append(array, [img], axis=0)
            i += 1
        return array


# In[5]:


def _make_imagedata(file_path_0, file_path_1, flat, color):
    return np.append(_convert_numpy(file_path_0, flat, color), _convert_numpy(file_path_1, flat, color), axis=0)


# In[6]:


def _make_labeldata(file_path_0, file_path_1):
    size_0 = len(glob.glob(file_path_0 + './*.jpg'))
    size_1 = len(glob.glob(file_path_1 + './*.jpg'))
    return np.append(np.zeros(size_0), np.ones(size_1))


# In[7]:


def read_image(normalize=True, flat=True, color=True):
    dataset = {}
    dataset['train_image'] = _make_imagedata(train_path_0, train_path_1, flat, color)
    dataset['train_label'] = _make_labeldata(train_path_0, train_path_1)
    dataset['test_image'] = _make_imagedata(test_path_0, test_path_1, flat, color)
    dataset['test_label'] = _make_labeldata(test_path_0, test_path_1)
    
    if normalize:
        for key in ('train_image', 'test_image'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    return (dataset['train_image'], dataset['train_label']), (dataset['test_image'], dataset['test_label'])


# In[8]:


if __name__ == '__main__':
    convertPNG2JPG(path_list)
    (x_train, t_train), (x_test, t_test) = read_image()


# In[9]:


if __name__ == '__main__':
    print(np.shape(x_train), np.shape(t_train), np.sum(t_train))
    print(np.shape(x_test), np.shape(t_test), np.sum(t_test))


# In[ ]:




