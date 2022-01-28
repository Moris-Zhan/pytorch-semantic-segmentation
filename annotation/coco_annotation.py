import os
import random
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import math
from pycocotools.coco import COCO
import torch
import collections
#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'D://WorkSpace/JupyterWorkSpace/DataSet/COCO'
VOCdevkit_sets  = [('train'), ('val')]

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
    total_dict = {}

    for year in ["2014", "2017"]:
    # for year in ["2017"]:
        for image_set in VOCdevkit_sets:
            mask_folder = "{}\\mask\\{}{}\\".format(VOCdevkit_path, image_set, year)
           
            if not os.path.exists(mask_folder): os.makedirs(mask_folder)
  
            segfilepath     = os.path.join(VOCdevkit_path, 'mask/%s%s'%(image_set, year))
            saveBasePath    = os.path.join(VOCdevkit_path, 'Segmentation')
            dir_ = os.path.join(VOCdevkit_path, "Segmentation")
            if not os.path.exists(dir_): os.makedirs(dir_)
            
            temp_seg = os.listdir(segfilepath)
            # total_seg = []
            # for seg in temp_seg:
            #     if seg.endswith(".png"):
            #         total_seg.append(seg)
            total_seg = glob(segfilepath + "/*.png")
            # total_dict = {i:1 for i in range(90 + 1)}
            
            for seg in tqdm(total_seg):
                img        = cv2.imread(seg, 0)
                # img = img[:,:,[2,1,0]]
                # img = encode_segmap(img)
                unique, counts = np.unique(img, return_counts=True)

                d = dict(zip(unique, counts)) 
                for k, v in d.items():
                    if k not in total_dict.keys(): total_dict[k] = 0
                    total_dict[k] += d[k]                                  

            num     = len(total_seg)  
            list    = range(num)              
            ftotal        = open(os.path.join(VOCdevkit_path, "Segmentation",'%s%s.txt'%(image_set, year)), 'w')  
            
            for i  in list:  
                name=total_seg[i][:-4]+'\n'  
                if i in list: 
                    mask = name.strip()
                    img = name
                    img = name.replace("mask", "images").strip()
                    ftotal.write(img + " " + mask + "\n") 
                  
            
            ftotal.close()           
            print("Generate txt in ImageSets done.")

    w      = open(os.path.join(dir_,'weight.txt'), 'w') 
    od = collections.OrderedDict(sorted(total_dict.items()))
    res = create_class_weight(od)
    [w.write(str(we)+"\n") for we in res.items()]

    w.close()
    print(" weight:", res.values()) 


    # merge to txt
    for image_set in VOCdevkit_sets:  
        txts = glob('%s/%s/%s*.txt'%(VOCdevkit_path, "Segmentation", image_set))
        # Reading data from file1
        with open(txts[0]) as fp:
            data = fp.read()
        
        # Reading data from file2
        with open(txts[1]) as fp:
            data2 = fp.read()

        # Merging 2 files
        # To add the data of file2
        # from next line
        data += "\n"
        data += data2
        
        with open(os.path.join(VOCdevkit_path, "Segmentation", "%s.txt"%(image_set)), 'w') as fp:
            fp.write(data)
