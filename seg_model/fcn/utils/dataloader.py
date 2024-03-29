import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from seg_model.fcn.utils.utils import preprocess_input, cvtColor
from seg_model.fcn.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import torch


class FCNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(FCNDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        self.albumentations = Albumentations(self.input_shape, self.train) 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#       
        # jpg         = cv2.imread(name.replace("mask_annotations", "images") + ".jpg")     
        # png         = cv2.imread(name.replace("mask_annotations", "mask_annotations") + ".png", 0)  # new mask weight image  

        jpg         = cv2.imread(annotation_line.split()[0] + ".jpg")     
        png         = cv2.imread(annotation_line.split()[1] + ".png", 0)  # new mask weight image  
                                    
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        # Albumentations
        # jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
        jpg, png    = self.albumentations(jpg, png) 
        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])

        png         = np.array(png)          
        png[png >= self.num_classes] = self.num_classes        

        # unique, counts = np.unique(png, return_counts=True)
        # d = dict(zip(unique, counts)) 
        # print(d)
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels
    
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
    #     image = cvtColor(image)
    #     label = Image.fromarray(np.array(label))
    #     # print("unique3:", np.unique(label)) 
    #     h, w = input_shape

    #     if not random:
    #         iw, ih  = image.size
    #         scale   = min(w/iw, h/ih)
    #         nw      = int(iw*scale)
    #         nh      = int(ih*scale)

    #         image       = image.resize((nw,nh), Image.BICUBIC)
    #         new_image   = Image.new('RGB', [w, h], (128,128,128))
    #         new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    #         label       = label.resize((nw,nh), Image.NEAREST)
    #         new_label   = Image.new('L', [w, h], (0))
    #         new_label.paste(label, ((w-nw)//2, (h-nh)//2))
    #         # print("unique_r:", np.unique(new_label)) 
    #         return new_image, new_label

    #     # resize image
    #     rand_jit1 = self.rand(1-jitter,1+jitter)
    #     rand_jit2 = self.rand(1-jitter,1+jitter)
    #     new_ar = w/h * rand_jit1/rand_jit2

    #     scale = self.rand(0.25, 2)
    #     if new_ar < 1:
    #         nh = int(scale*h)
    #         nw = int(nh*new_ar)
    #     else:
    #         nw = int(scale*w)
    #         nh = int(nw/new_ar)

    #     image = image.resize((nw,nh), Image.BICUBIC)
    #     label = label.resize((nw,nh), Image.NEAREST)
    #     # print("unique4:", np.unique(label)) 
        
    #     flip = self.rand()<.5
    #     if flip: 
    #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #         label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #     unique5 = np.unique(label)
    #     # print("unique5:", unique5) 

    #     # place image
    #     dx = int(self.rand(0, w-nw))
    #     dy = int(self.rand(0, h-nh))
    #     new_image = Image.new('RGB', (w,h), (128,128,128))
    #     new_label = Image.new('L', (w,h), (0))
    #     new_image.paste(image, (dx, dy))
    #     new_label.paste(label, (dx, dy))
        
    #     unique6 = np.unique(new_label)
    #     # print("unique6:", unique6) 
    #     # if(unique5.size != unique6.size):
    #     #     new_image.show()
    #     #     # label.show()
    #     #     new_label.show()
    #     #     print()

    #     image = new_image
    #     label = new_label

    #     # distort image
    #     hue = self.rand(-hue, hue)
    #     sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
    #     val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
    #     x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    #     x[..., 0] += hue*360
    #     x[..., 0][x[..., 0]>1] -= 1
    #     x[..., 0][x[..., 0]<0] += 1
    #     x[..., 1] *= sat
    #     x[..., 2] *= val
    #     x[x[:,:, 0]>360, 0] = 360
    #     x[:, :, 1:][x[:, :, 1:]>1] = 1
    #     x[x<0] = 0
    #     image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
    #     return image_data,label


# DataLoader中collate_fn使用
def fcn_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels