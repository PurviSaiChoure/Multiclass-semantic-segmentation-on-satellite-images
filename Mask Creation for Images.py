#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import matplotlib.pyplot as plt

coco_annotation_file = 'Downloads/seg/result.json'
mask_save_dir = 'Downloads/seg/masks'

os.makedirs(mask_save_dir, exist_ok=True)

coco = COCO(coco_annotation_file)
image_ids = coco.getImgIds()


category_mapping = {0: 0, 1: 1, 2: 2, 3: 3}

categories = coco.loadCats(coco.getCatIds())
print("Categories in COCO file:", categories)

for idx, image_id in enumerate(sorted(image_ids), start=1):
    image_info = coco.loadImgs(image_id)[0]
    height, width = image_info['height'], image_info['width']
    print(f"Processing image: {image_info['file_name']} (ID: {image_id})")
    mask = np.zeros((height, width), dtype=np.uint8)
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    
    for ann in annotations:
        category_id = ann['category_id']
        class_index = category_mapping.get(category_id, -1)
        
        print(f"Annotation category_id: {category_id}, class_index: {class_index}")  
        
        if class_index == -1:
            continue
        
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):  
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    cv2.fillPoly(mask, [poly.astype(np.int32)], class_index)
            elif isinstance(ann['segmentation'], dict):  
                rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
                m = maskUtils.decode(rle)
                mask[m == 1] = class_index
            elif isinstance(ann['segmentation'], str):  
                rle = maskUtils.frPyObjects([ann['segmentation']], height, width)
                m = maskUtils.decode(rle)
                mask[m == 1] = class_index
    mask_filename = f"{idx}.png"
    mask_path = os.path.join(mask_save_dir, mask_filename)
    cv2.imwrite(mask_path, mask)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Mask for image {idx}')
    plt.show()
    if os.path.exists(mask_path):
        print(f'Mask saved at: {mask_path}')
    else:
        print(f'Error saving mask at: {mask_path}')

    unique_values = np.unique(mask)
    print(f"Unique values in the mask for image {image_info['file_name']}: {unique_values}")

