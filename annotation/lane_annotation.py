import os
import random
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from concurrent.futures import ThreadPoolExecutor
import math
#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
# train_percent       = 0.9
train_percent       = 1
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'D://WorkSpace//JupyterWorkSpace//DataSet//LANEdevkit'

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
            (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_mask_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask

def get_mask_labels():
    unlabeled = [0, 0, 0]
    BL=[0,255,255]
    CL=[0,128,255]
    DM=[178,102,255]
    JB=[255,255,51]
    LA=[255,102,178]
    PC=[255,255,0]
    RA=[255,0,127]
    SA=[255,0,255]
    SL=[0,255,0]
    SLA=[255,128,0]
    SRA=[255,0,0]

    return np.array([
        unlabeled, BL, CL, DM, JB, LA, PC, RA, SA, SL, SLA, SRA])

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum([labels_dict[key] for key in labels_dict])
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
  
    saveBasePath    = os.path.join(VOCdevkit_path, 'Segmentation')
    if not os.path.exists(saveBasePath): os.makedirs(saveBasePath)

    
    for imgSet in ["train", "test"]:
        segfilepath     = os.path.join(VOCdevkit_path, '%s//mask_annotations(orign)'%(imgSet), "*.png")

        saveSegPath    = os.path.join(VOCdevkit_path, '%s//mask_annotations'%(imgSet))
        if not os.path.exists(saveSegPath): os.makedirs(saveSegPath)
     
        total_seg = glob(segfilepath)
        num     = len(total_seg)  
        list    = range(num) 
     
        print("%s size"%(imgSet),num) 

        if imgSet == "test":
            file      = open(os.path.join(saveBasePath,'val.txt'), 'w') 
        else:
            file      = open(os.path.join(saveBasePath,'%s.txt'%(imgSet)), 'w') 

        # total_dict = {i:1 for i in range(11 + 1)}
        total_dict = {}

        with tqdm(total=len(list)) as pbar:
            with ThreadPoolExecutor(max_workers=8) as ex:
                for i in list: 
                    name=total_seg[i][:-4]  
                    data        = cv2.imread(name + ".png", cv2.COLOR_BGR2RGB)
                    data = data[:,:,[2,1,0]]
                    data = encode_segmap(data)
                    unique, counts = np.unique(data, return_counts=True)
                    d = dict(zip(unique, counts)) 
                    for k, v in d.items():
                        if k not in total_dict.keys(): total_dict[k] = 0
                        total_dict[k] += d[k]
                    
                    mask = name.replace("mask_annotations(orign)", "mask_annotations")
                    img = mask.replace("mask_annotations", "images")

                    file.write(img + " " + mask +'\n')  # write new mask path                   
                    cv2.imwrite(mask + ".png", data) # export new mask img                     
                    pbar.update(1)
    
        file.close()  
 
        w      = open(os.path.join(saveBasePath,'weight_%s.txt'%(imgSet)), 'w') 
        res = create_class_weight(total_dict)
        [w.write(str(we)+"\n") for we in res.items()]

        w.close()
        print("%s weight:"%(imgSet), res.values())
        print("Generate txt in ImageSets done.")
