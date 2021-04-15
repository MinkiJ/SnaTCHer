import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, 'data/cub')
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

data_path = "/media/personal-k/2.0TB/datasets/CUB_200_2011"

# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)

import os

class CUB(Dataset):

    def __init__(self, setname, args, augment=False):
        
        with open(os.path.join(data_path, 'images.txt'), 'r') as f:
          image_list = f.readlines()
        
        image_index = []
        image_path = []
        for data in image_list:
          index, path = data.split(' ')
          image_index.append(int(index))
          image_path.append(os.path.join(data_path, 'images', path[:-1]))
        
        self.image_path = image_path
        
        train_flag = np.loadtxt(os.path.join(data_path, 'train_test_split.txt'), delimiter=' ', dtype=np.int32)
        train_flag = train_flag[:, 1]
        labels = np.loadtxt(os.path.join(data_path, 'image_class_labels.txt'), delimiter=' ', dtype=np.int32)
        labels = labels[:, 1]
        
        # use first 100 classes
        targets = np.where(labels < 101)[0]
        self.labels = labels
        self.indices = targets
        self.label = list(self.labels[self.indices] - 1)
        self.num_classes = self.num_class = 100

        image_size = 84
        
        if augment and setname == 'train':
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
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])            
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])         
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        path = self.image_path[index]; label = self.labels[index]
        image = self.transform(Image.open(path).convert('RGB'))
        
        return image, label            

