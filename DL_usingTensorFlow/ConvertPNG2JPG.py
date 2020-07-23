#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import sys
import os
import re
import glob


# In[4]:


def convertPNG2JPG(file_path):
    count = 1
    for path in file_path:
        files = glob.glob(path + './*.png')
        #print(1)
        for file in files:
            input_im = Image.open(file, 'r')
            rgb_im = input_im.convert('RGB')
            rgb_im.save(file + 'R' + '.jpg', 'JPEG')
            os.remove(file)
            count = count + 1
            print("transcation finished" + str(count))


# In[3]:


if __name__ == '__main__':
    input_path = os.getcwd() + '\\'
    output_path = os.getcwd() + '\\' 
    files = os.listdir(os.getcwd() + '\\')
    train_path_0 = os.getcwd() + '\\' + 'train_images' + '\\' + 'not_kumamon'
    train_path_1 = os.getcwd() + '\\' + 'train_images' + '\\' + 'kumamon'
    test_path_0 = os.getcwd() + '\\' + 'test_images' + '\\' + 'not_kumamon'
    test_path_1 = os.getcwd() + '\\' + 'test_images' + '\\' + 'kumamon'
    
    file_path = [train_path_0, train_path_1, test_path_0, test_path_1]
    
    convertPNG2JPG(file_path)

