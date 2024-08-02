#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import glob

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU
import random


# In[2]:


temp_img = cv2.imread("Downloads/seg/images/1.jpg") 


# In[3]:


plt.imshow(temp_img[:,:,0])


# In[7]:


temp_mask = cv2.imread("Downloads/seg/masks/8.png")  
labels, count = np.unique(temp_mask[:,:,0], return_counts=True) 
print("Labels are: ", labels, " and the counts are: ", count)


# In[8]:


import os
import cv2
import numpy as np

mask_dir = "Downloads/seg/masks/"

mask_files = os.listdir(mask_dir)
unique_values_set = set()
for mask_file in mask_files:
    mask_path = os.path.join(mask_dir, mask_file)
    temp_mask = cv2.imread(mask_path)
    unique_values_set.update(np.unique(temp_mask))

num_classes = len(unique_values_set)
print("Maximum number of classes across all mask images:", num_classes)


# In[9]:


import matplotlib.pyplot as plt

plt.imshow(temp_mask[:,:,0], cmap='gray')
plt.title('Segmentation Mask')
plt.show()


# In[10]:


import cv2
import numpy as np

temp_mask = cv2.imread("Downloads/seg/masks/15.png", cv2.IMREAD_GRAYSCALE)
print(temp_mask.shape)


# In[11]:


temp_mask = cv2.imread("Downloads/seg/masks/22.png") 
labels, count = np.unique(temp_mask[:,:,0], return_counts=True) 
print("Labels are: ", labels, " and the counts are: ", count)


# In[12]:


root_directory = 'Downloads/seg'

patch_size = 256 
img_dir=root_directory+"images"
for path, subdirs, files in os.walk(img_dir): 
    dirname = path.split(os.path.sep)[-1]
    #print(dirname)
    images = os.listdir(path)
    for i, image_name in enumerate(images):  
        if image_name.endswith(".jpg"):
            #print(image_name)
            image = cv2.imread(path+"/"+image_name, 1)  
            SIZE_X = (image.shape[1]//patch_size)*patch_size 
            SIZE_Y = (image.shape[0]//patch_size)*patch_size 
            image = Image.fromarray(image)
            image = image.crop((0 ,0, SIZE_X, SIZE_Y)) 
            image = np.array(image)             
   
            print("Now patchifying image:", path+"/"+image_name)
            patches_img = patchify(image, (256, 256, 3), step=256) 
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = single_patch_img[0] 
                    cv2.imwrite(root_directory+"256_patches/images/"+
                               image_name+"patch_"+str(i)+str(j)+".jpg", single_patch_img)
            
mask_dir=root_directory+"masks"
for path, subdirs, files in os.walk(mask_dir): 
    dirname = path.split(os.path.sep)[-1]

    masks = os.listdir(path)  
    for i, mask_name in enumerate(masks):  
        if mask_name.endswith(".png"):           
            mask = cv2.imread(path+"/"+mask_name, 0)  
            SIZE_X = (mask.shape[1]//patch_size)*patch_size 
            SIZE_Y = (mask.shape[0]//patch_size)*patch_size 
            mask = Image.fromarray(mask)
            mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  
            mask = np.array(mask)             
            print("Now patchifying mask:", path+"/"+mask_name)
            patches_mask = patchify(mask, (256, 256), step=256)  
    
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    
                    single_patch_mask = patches_mask[i,j,:,:]                         
                    cv2.imwrite(root_directory+"256_patches/masks/"+
                               mask_name+"patch_"+str(i)+str(j)+".png", single_patch_mask)


# In[13]:


root_directory = 'Downloads/seg/'
patch_size = 256
img_dir = root_directory + "images/"
output_dir = root_directory + "256_patches/images/"
os.makedirs(output_dir, exist_ok=True)

for path, subdirs, files in os.walk(img_dir):
    images = [f for f in files if f.endswith(".jpg")]
    
    for i, image_name in enumerate(images):
        print(image_name)
        image = cv2.imread(os.path.join(path, image_name), 1)
        
        if image is None:
            continue  
        
        SIZE_X = (image.shape[1] // patch_size) * patch_size  
        SIZE_Y = (image.shape[0] // patch_size) * patch_size  
        
        image = Image.fromarray(image)
        image = image.crop((0, 0, SIZE_X, SIZE_Y))  
        image = np.array(image)
        print(f"Now patchifying image: {os.path.join(path, image_name)}")
        patches_img = patchify(image, (256, 256, 3), step=256)  
        
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, 0, :, :, :]  
                save_path = os.path.join(output_dir, f"{image_name}_patch_{i}_{j}.jpg")
                cv2.imwrite(save_path, single_patch_img)


# In[14]:


mask_dir = root_directory + "masks"
output_mask_dir = root_directory + "256_patches/masks/"
os.makedirs(output_mask_dir, exist_ok=True)

for path, subdirs, files in os.walk(mask_dir):
    masks = [f for f in files if f.endswith(".png")]
    
    for i, mask_name in enumerate(masks):
        print(mask_name)
        mask = cv2.imread(os.path.join(path, mask_name), 0) 
        
        if mask is None:
            continue  
        
        SIZE_X = (mask.shape[1] // patch_size) * patch_size  
        SIZE_Y = (mask.shape[0] // patch_size) * patch_size  
        
        mask = Image.fromarray(mask)
        mask = mask.crop((0, 0, SIZE_X, SIZE_Y))  
        mask = np.array(mask)
        print(f"Now patchifying mask: {os.path.join(path, mask_name)}")
        patches_mask = patchify(mask, (256, 256), step=256)  
        
        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]  
                save_path = os.path.join(output_mask_dir, f"{mask_name}_patch_{i}_{j}.png")
                cv2.imwrite(save_path, single_patch_mask)


# In[15]:


train_img_dir = "Downloads/seg/256_patches/images/"
train_mask_dir = "Downloads/seg/256_patches/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))


img_num = random.randint(0, num_images-1)

img_for_plot = cv2.imread(train_img_dir+img_list[img_num], 1)
img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

mask_for_plot =cv2.imread(train_mask_dir+msk_list[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot)
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()


# In[16]:


useless=0  
for img in range(len(img_list)):   
    img_name=img_list[img]
    mask_name = msk_list[img]
    print("Now preparing image and masks number: ", img)
      
    temp_image=cv2.imread(train_img_dir+img_list[img], 1)
   
    temp_mask=cv2.imread(train_mask_dir+msk_list[img], 0)
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.05:  
        print("Save Me")
        cv2.imwrite('Downloads/seg/256_patches/images_with_useful_info/images/'+img_name, temp_image)
        cv2.imwrite('Downloads/seg/256_patches/images_with_useful_info/masks/'+mask_name, temp_mask)
        
    else:
        print("I am useless")   
        useless +=1

print("Total useful images are: ", len(img_list)-useless)  
print("Total useless images are: ", useless) 


# In[17]:


import os
import cv2
import numpy as np

useless = 0
train_img_dir = "Downloads/seg/256_patches/images/"
train_mask_dir = "Downloads/seg/256_patches/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

output_img_dir = 'Downloads/seg/256_patches/images_with_useful_info/images/'
output_mask_dir = 'Downloads/seg/256_patches/images_with_useful_info/masks/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

for img in range(len(img_list)):
    img_name = img_list[img]
    mask_name = msk_list[img]
    print("Now preparing image and masks number: ", img)
    
    temp_image = cv2.imread(os.path.join(train_img_dir, img_name), 1)
    temp_mask = cv2.imread(os.path.join(train_mask_dir, mask_name), 0)
    
    if temp_image is None or temp_mask is None:
        print(f"Skipping {img_name} or {mask_name} as they couldn't be loaded.")
        continue
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0] / counts.sum())) > 0.05:
        print("Save Me")
        cv2.imwrite(os.path.join(output_img_dir, img_name), temp_image)
        cv2.imwrite(os.path.join(output_mask_dir, mask_name), temp_mask)
    else:
        print("I am useless")
        useless += 1

print("Total useful images are: ", len(img_list) - useless)
print("Total useless images are: ", useless)


# In[18]:


import splitfolders


# In[19]:


input_folder = 'Downloads/seg/256_patches/images_with_useful_info/'
output_folder = 'Downloads/seg/data_for_training_and_testing/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) 






