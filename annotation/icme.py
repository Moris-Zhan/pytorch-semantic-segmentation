import os
import random
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from concurrent.futures import ThreadPoolExecutor
import math
import shutil 
#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9


# def encode_segmap(mask):
#     """Encode segmentation label images as pascal classes
#     Args:
#         mask (np.ndarray): raw segmentation label image of dimension
#             (M, N, 3), in which the Pascal classes are encoded as colours.
#     Returns:
#         (np.ndarray): class map with dimensions (M,N), where the value at
#         a given location is the integer denoting the class index.
#     """
#     mask = mask.astype(int)
#     label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
#     for ii, label in enumerate(get_mask_labels()):
#         label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
#     label_mask = label_mask.astype(int)
#     return label_mask

# def get_mask_labels():
#     unlabeled = [0, 0, 0]
#     BL=[0,255,255]
#     CL=[0,128,255]
#     DM=[178,102,255]
#     JB=[255,255,51]
#     LA=[255,102,178]
#     PC=[255,255,0]
#     RA=[255,0,127]
#     SA=[255,0,255]
#     SL=[0,255,0]
#     SLA=[255,128,0]
#     SRA=[255,0,0]

#     return np.array([
#         unlabeled, BL, CL, DM, JB, LA, PC, RA, SA, SL, SLA, SRA])

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum([labels_dict[key] for key in labels_dict])
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

def get_annotation(data_root):
    random.seed(0)
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = os.path.join(data_root, "ICME2022")
    print("Generate txt in ImageSets.")
  
    saveBasePath    = os.path.join(VOCdevkit_path, 'Segmentation')
    if not os.path.exists(saveBasePath): os.makedirs(saveBasePath)

    segfilepath     = os.path.join(VOCdevkit_path, 'labels//class_labels', "*.png")
    total_seg = glob(segfilepath)  
    # total_dict = {i:1 for i in range(5 + 1)}
    total_dict = {}

    for seg in tqdm(total_seg):       
        data        = cv2.imread(seg, 0)
        # data = data[:,:,[2,1,0]]
        # data = encode_segmap(data)
        unique, counts = np.unique(data, return_counts=True)

        d = dict(zip(unique, counts)) 
        for k, v in d.items():
            if k not in total_dict.keys(): total_dict[k] = 0
            total_dict[k] += d[k]
    
    w      = open(os.path.join(saveBasePath,'weight.txt'), 'w') 
    res = create_class_weight(total_dict)
    [w.write(str(we)+"\n") for we in res.items()]

    num     = len(total_seg)  
    list    = range(num) 
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr) 

    print("train and val size",tv)
    print("train size",tr)
    print("val size",tv-tr)
   
    ftrain      = open(os.path.join(VOCdevkit_path, "Segmentation",'train.txt'), 'w')  
    fval        = open(os.path.join(VOCdevkit_path, "Segmentation",'val.txt'), 'w')

    for i  in list:  
        name=total_seg[i][:-4]+'\n'  
        if i in trainval:  
            # ftrainval.write(name)  
            if i in train:  
                mask = name.strip()
                img = name.replace("labels//class_labels", "images").replace("_lane_line_label_id", "").strip()
                ftrain.write(img + " " + mask + "\n")  
            else:  
                mask = name.strip()
                img = name.replace("labels//class_labels", "images").replace("_lane_line_label_id", "").strip()
                fval.write(img + " " + mask + "\n")   
        # else:  
        #     ftest.write(name)  
    
    # ftrainval.close()  
    ftrain.close()  
    fval.close()  
    # ftest.close()
    print("Generate txt in ImageSets done.")   
   
if __name__ == "__main__":
    data_root = '/home/leyan/DataSet/'
    get_annotation(data_root)   