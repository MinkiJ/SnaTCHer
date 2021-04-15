#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:29:35 2021

@author: personal-k
"""

import torch

import glob
import os
import pickle
import torchvision.transforms as transforms
import numpy as np

from PIL import Image


cub_root = "/media/personal-k/2.0TB/datasets/CUB_200_2011/images"
cub_split = "/home/personal-k/fewshot/FEAT/meta-dataset/cub_class_split.pkl"
imagenet_root = "/home/personal-k/datasets/ILSVRC2012_img_train"
imagenet_split = "/home/personal-k/fewshot/FEAT/meta-dataset/imagenet_class_split.pkl"
duplicate_string = './meta-dataset/ImageNet_%s_duplicates.txt'


class MetaDataset(torch.utils.data.Dataset):
  def __init__(self, mode,
               imagenet_root="/home/personal-k/datasets/ILSVRC2012_img_train",
               imagenet_split = "./meta-dataset/imagenet_class_split.pkl",
               cub_root = "/media/personal-k/2.0TB/datasets/CUB_200_2011/images",
               cub_split = "./meta-dataset/cub_class_split.pkl",
               duplicate_string = './meta-dataset/ImageNet_%s_duplicates.txt',
               ):
    super().__init__()
    self.imagenet_root = imagenet_root
    self.imagenet_split = imagenet_split
    self.cub_root = cub_root
    self.cub_split = cub_split
    self.mode = mode

    image_size = 84
    if 'train' in mode:
        transforms_list = [
              transforms.RandomResizedCrop(image_size),
              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
            ]
    else:
        transforms_list = [
              transforms.Resize(92),
              transforms.CenterCrop(image_size),
              transforms.ToTensor(),
            ]

    # Transformation
#    if args.backbone_class == 'ConvNet':
#        self.transform = transforms.Compose(
#            transforms_list + [
#            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
#                                 np.array([0.229, 0.224, 0.225]))
#        ])
#    elif args.backbone_class == 'Res12':
    self.transform = transforms.Compose(
        transforms_list + [
        transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                             np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
    ])
    
    self._init_filelist(imagenet_root, imagenet_split, cub_root, cub_split,
                        duplicate_string)
    
    if mode == 'imagenet_train':
      self.filelist = self.imagenet_train_files
    elif mode == 'imagenet_val':
      self.filelist = self.imagenet_val_files
    elif mode == 'imagenet_test':
      self.filelist = self.imagenet_test_files
    elif mode == 'cub_test':
      self.filelist = self.cub_test_files
    elif mode == 'cub_train':
      self.filelist = self.cub_train_files
    elif mode == 'cub_val':
      self.filelist = self.cub_val_files
    
    self.num_classwise_images = [len(f) for f in self.filelist]
    self.num_images = sum(self.num_classwise_images)
    self.index_filelist = []
    self.index_labellist = []
    for i, f in enumerate(self.filelist):
      self.index_filelist += f
      self.index_labellist += [i] * len(f)
    
  def __len__(self):
    return self.num_images
  
  def __getitem__(self, index):
    filepath = self.index_filelist[index]
    label = self.index_labellist[index]
    
    image = Image.open(filepath).convert('RGB')
    
    if self.transform is not None:
      image = self.transform(image)
    
    return image, label
    
  
  def _init_filelist(self, imagenet_root, imagenet_split, cub_root, cub_split,
                     duplicate_string):
    with open(cub_split, 'rb') as f:
      cub_split = pickle.load(f)
    with open(imagenet_split, 'rb') as f:
      imagenet_split = pickle.load(f)
    
    cub_train_files = [glob.glob(os.path.join(cub_root, t, '*')) for t in cub_split['train']]
    cub_val_files = [glob.glob(os.path.join(cub_root, t, '*')) for t in cub_split['val']]
    cub_test_files = [glob.glob(os.path.join(cub_root, t, '*')) for t in cub_split['test']]
    
    imagenet_train_files = [glob.glob(os.path.join(imagenet_root, t, '*')) for t in imagenet_split['train']]
    imagenet_val_files = [glob.glob(os.path.join(imagenet_root, t, '*')) for t in imagenet_split['val']]
    imagenet_test_files = [glob.glob(os.path.join(imagenet_root, t, '*')) for t in imagenet_split['test']]
    
    """ remove duplicates from imagenet """
    #for target_dataset in ('Caltech101', 'Caltech256', 'CUBirds'):
    for target_dataset in ( 'CUBirds', 'Caltech101', 'Caltech256'):
      with open(duplicate_string % target_dataset, 'r') as f:
        duplicates = f.readlines()
      
      duplicate_count = 0
      target_count = []
      for data in duplicates:
        if data.startswith('#'):
          continue
        target = data.split('.JPEG')[0] + '.JPEG'
        class_name = target.split('/')[0]
        
        target_count.append(target)
        
        if class_name in imagenet_split['train']:
          cls_index = imagenet_split['train'].index(class_name)
          for i in range(len(imagenet_train_files[cls_index])):
            if imagenet_train_files[cls_index][i].endswith(target):
              del imagenet_train_files[cls_index][i]
              duplicate_count += 1
              break
        elif class_name in imagenet_split['val']:
          cls_index = imagenet_split['val'].index(class_name)
          for i in range(len(imagenet_val_files[cls_index])):
            if imagenet_val_files[cls_index][i].endswith(target):
              del imagenet_val_files[cls_index][i]
              duplicate_count += 1
              break
        elif class_name in imagenet_split['test']:
          cls_index = imagenet_split['test'].index(class_name)
          for i in range(len(imagenet_test_files[cls_index])):
            if imagenet_test_files[cls_index][i].endswith(target):
              del imagenet_test_files[cls_index][i]
              duplicate_count += 1
              break
        else:
          print('Not in data', class_name)
      
    self.imagenet_train_files = imagenet_train_files
    self.imagenet_val_files = imagenet_val_files
    self.imagenet_test_files = imagenet_test_files
    
    self.cub_train_files = cub_train_files
    self.cub_val_files = cub_val_files
    self.cub_test_files = cub_test_files
    
    print('Filelist Init Done')




