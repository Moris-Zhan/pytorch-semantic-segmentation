import numpy as np
import os
import math
import torch
import torch.nn as nn
from copy import deepcopy

from pynvml import *
from functools import partial
import threading

#---------------------------------------#
#   获得学习率下降的公式
#---------------------------------------#
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)    

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

# def get_data_path(root_path, dataType):
#     #------------------------------#  
#     #   數據集路徑
#     #   訓練自己的數據集必須要修改的
#     #------------------------------#  
#     map_dict = { "icme":'ICME2022', "coco":'COCO',
#                  "voc":'VOCdevkit', "lane":"LANEdevkit", "cityscape":'Cityscapes'}
#     return os.path.join(root_path, map_dict[dataType]) 


class LossHistory():
    def __init__(self, opt, patience = 10):        
        self.losses     = []
        self.val_loss   = []
        self.writer = opt.writer
        self.freeze = False
        self.log_dir = opt.out_path
       
        if opt.local_rank == 0:
            # launch tensorboard
            t = threading.Thread(target=self.launchTensorBoard, args=([opt.out_path]))
            t.start()       

        # initial EarlyStopping
        self.patience = patience
        self.reset_stop()          

    def launchTensorBoard(self, tensorBoardPath, port = 8888):
        os.system('tensorboard --logdir=%s --port=%s --load_fast=false'%(tensorBoardPath, port))
        url = "http://localhost:%s/"%(port)
        # webbrowser.open_new(url)
        return

    def reset_stop(self):
        self.best_epoch_loss = np.Inf 
        self.stopping = False
        self.counter  = 0

    def set_status(self, freeze):
        self.freeze = freeze

    def epoch_loss(self, loss, val_loss, epoch):
        self.losses.append(loss)
        self.val_loss.append(val_loss)  

        prefix = "Freeze_epoch/" if self.freeze else "UnFreeze_epoch/"     
        self.writer.add_scalar(prefix+'Loss/Train', loss, epoch)
        self.writer.add_scalar(prefix+'Loss/Val', val_loss, epoch)
        self.decide(val_loss)   

    def step(self, steploss, iteration):        
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Loss', steploss, iteration)

    def step_c(self, steploss, iteration):  # for centernet      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Classification_Loss', steploss, iteration)

    def step_r(self, steploss, iteration):  # for centernet        
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Regression_Loss', steploss, iteration)

    def step_rpn_loc(self, steploss, iteration):  # for fasterrcnn      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/rpn_loc_Loss', steploss, iteration)

    def step_rpn_cls(self, steploss, iteration):  # for fasterrcnn      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/rpn_cls_Loss', steploss, iteration)

    def step_roi_loc(self, steploss, iteration):  # for fasterrcnn      
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/roi_loc_Loss', steploss, iteration)

    def step_roi_cls(self, steploss, iteration):  # for fasterrcnn     
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/roi_cls_Loss', steploss, iteration)        

    def decide(self, epoch_loss):
        if epoch_loss > self.best_epoch_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'Best lower loss:{self.best_epoch_loss}')
                self.stopping = True
        else:
            self.best_epoch_loss = epoch_loss           
            self.counter = 0 
            self.stopping = False

def get_cls_weight(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [float(line.split(", ")[1].split(")")[0]) for line in lines]
        cls_weights     = np.array(lines, np.float32)
    return cls_weights

def get_data(root_path, dataType):
    map_dict = { "icme":'ICME2022', "coco":'COCO',
                 "voc":'VOCdevkit', "lane":"LANEdevkit", "cityscape":'Cityscapes'}
    VOCdevkit_path = os.path.join(root_path, map_dict[dataType]) 

    if dataType == "voc":
        #   VOCdevkit
        VOCdevkit_path  = os.path.join(root_path, "VOCdevkit")    
        num_classes = 20 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))

        name_classes    = ["background", "aeroplane", "tvmonitor", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 'diningtable', 
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train"]

    elif dataType == "lane":
        #   LANEdevkit
        VOCdevkit_path  = os.path.join(root_path, "LANEdevkit")    
        num_classes = 11 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight_train.txt" %(VOCdevkit_path))
        name_classes    = ["background", "BL", "CL", "DM", "JB", "LA", "PC", "RA", "SA", "SL", "SLA", "SRA"]

    elif dataType == "icme":
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "ICME2022")    
        num_classes = 5 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))  
        name_classes    = ["background", "main_lane", "alter_lane", "double_line", "dashed_line", "single_line"]

    elif dataType == "coco":
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "COCO")    
        num_classes = 80 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))    
        name_classes    = ["background", "person", "bicycle", "car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
                            "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
                            "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
                            "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
                            "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

    elif dataType == "cityscape":
        #   Cityscape
        VOCdevkit_path  = os.path.join(root_path, "Cityscapes")    
        num_classes = 19 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))    
        name_classes = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

    return VOCdevkit_path, num_classes, cls_weights, name_classes