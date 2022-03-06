#-------------------------------------#
#       對數據集進行訓練
#-------------------------------------#
import numpy as np
from pytorch_lightning import os
import torch
from torch.utils.data import DataLoader

# from yolov4.utils.dataloader import YoloDataset as Dataset , yolo_dataset_collate as dataset_collate
# from yolov4.utils.utils import get_classes

from seg_model.pspnet.nets.pspnet_training import weights_init
from seg_model.pspnet.utils.callbacks import LossHistory
from seg_model.pspnet.utils.utils_fit import fit_one_epoch
from seg_model.pspnet.utils.dataloader import PSPnetDataset, pspnet_dataset_collate

from tqdm import tqdm

class DataType:
    VOC   = 0
    LANE  = 1
    ICME  = 2  
    COCO  = 3
    Cityscape = 4

def get_cls_weight(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [float(line.split(", ")[1].split(")")[0]) for line in lines]
        cls_weights     = np.array(lines, np.float32)
    return cls_weights    

if __name__ == "__main__":
    # root_path = 'D://WorkSpace//JupyterWorkSpace//DataSet//bdd100k'
    root_path = "D://WorkSpace//JupyterWorkSpace//DataSet"
    dataType = DataType.Cityscape
    #------------------------------------------------------------------------------------#
    
    #-------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #-------------------------------#
    Cuda = True
    #--------------------------------------------------------#
    #   訓練前一定要修改classes_path，使其對應自己的數據集
    #--------------------------------------------------------#
    if dataType == DataType.VOC:
        #   VOCdevkit
        VOCdevkit_path  = os.path.join(root_path, "VOCdevkit")    
        num_classes = 20 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))
    elif dataType == DataType.LANE:
        #   LANEdevkit
        VOCdevkit_path  = os.path.join(root_path, "LANEdevkit")    
        num_classes = 11 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight_train.txt" %(VOCdevkit_path))
    elif dataType == DataType.ICME:
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "ICME2022")    
        num_classes = 5 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))  
    elif dataType == DataType.COCO:
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "COCO")    
        num_classes = 80 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))    
    elif dataType == DataType.Cityscape:
        #   Cityscape
        VOCdevkit_path  = os.path.join(root_path, "Cityscapes")    
        num_classes = 19 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))     
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #------------------------------------------------------#
    num_workers         = 4
    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "Segmentation//train.txt"),"r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "Segmentation//val.txt"),"r") as f:
        val_lines = f.readlines()
        
    #----------------------------------------------------#
    #   獲取classes和anchor
    #----------------------------------------------------
    # class_names, num_classes = get_classes(classes_path)   
    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    # with open(train_annotation_path) as f:
    #     train_lines = f.readlines()
    # with open(val_annotation_path) as f:
    #     val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    #------------------------------------------------------#
    batch_size  = 8
                    
    epoch_step      = num_train // batch_size

    input_shape     = [300, 300]

    # SSD
    train_dataset   = PSPnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset     = PSPnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                drop_last = True, collate_fn = pspnet_dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                drop_last = True, collate_fn = pspnet_dataset_collate)

    with tqdm(total=epoch_step,postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            imgs, pngs, labels = batch
            imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
            pngs    = torch.from_numpy(pngs).long()
            labels  = torch.from_numpy(labels).type(torch.FloatTensor)
            pbar.update(1)


    epoch_step      = num_val // batch_size

    with tqdm(total=epoch_step,postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            imgs, pngs, labels = batch
            imgs    = torch.from_numpy(imgs).type(torch.FloatTensor)
            pngs    = torch.from_numpy(pngs).long()
            labels  = torch.from_numpy(labels).type(torch.FloatTensor)
            pbar.update(1)