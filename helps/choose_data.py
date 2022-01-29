import os
import numpy as np


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

def get_data(root_path, dataType):
    #   數據集路徑
    #   訓練自己的數據集必須要修改的

    #   自己需要的分類個數+1，如2+1
    #   是否給不同種類賦予不同的損失權值，默認是平衡的。

    #   設置的話，注意設置成numpy形式的，長度和num_classes一樣。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------#  
    if dataType == DataType.VOC:
        #   VOCdevkit
        VOCdevkit_path  = os.path.join(root_path, "VOCdevkit")    
        num_classes = 20 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))

        name_classes    = ["background", "aeroplane", "tvmonitor", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 'diningtable', 
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train"]

    elif dataType == DataType.LANE:
        #   LANEdevkit
        VOCdevkit_path  = os.path.join(root_path, "LANEdevkit")    
        num_classes = 11 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight_train.txt" %(VOCdevkit_path))
        name_classes    = ["background", "BL", "CL", "DM", "JB", "LA", "PC", "RA", "SA", "SL", "SLA", "SRA"]

    elif dataType == DataType.ICME:
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "ICME2022")    
        num_classes = 5 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))  
        name_classes    = ["background", "main_lane", "alter_lane", "double_line", "dashed_line", "single_line"]

    elif dataType == DataType.COCO:
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

    elif dataType == DataType.Cityscape:
        #   Cityscape
        VOCdevkit_path  = os.path.join(root_path, "Cityscapes")    
        num_classes = 19 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))    
        name_classes = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

    return VOCdevkit_path, num_classes, cls_weights, name_classes