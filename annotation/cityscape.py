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
import collections
#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9


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
    return np.array([
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum([labels_dict[key] for key in labels_dict])
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight

def recursive_glob(rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

def get_annotation(data_root):
    random.seed(0)
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = os.path.join(data_root, "Cityscapes")
    VOCdevkit_sets  = [('train'), ('val'), ('test')]

    print("Generate txt in ImageSets.")
  
    saveBasePath    = os.path.join(VOCdevkit_path, 'Segmentation')
    if not os.path.exists(saveBasePath): os.makedirs(saveBasePath)

    NUM_CLASSES = 19 + 1 
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                        'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                        'motorcycle', 'bicycle']
    total_dict = {}

    for img_set in VOCdevkit_sets:
        files = {}
        images_base = os.path.join(VOCdevkit_path, 'leftImg8bit', img_set)
        annotations_base = os.path.join(VOCdevkit_path, 'gtFine_trainvaltest', 'gtFine', img_set) 
        # gtFine，顯然這裡的fine就是精細標註的意思。gtFine下面也是分為train， test以及val，然後它們的子目錄也是以城市為單位來放置圖片

        files[img_set] = recursive_glob(rootdir=images_base, suffix='.png')

        ftotal      = open(os.path.join(VOCdevkit_path, "Segmentation",'%s.txt'%img_set), 'w') 

        for i in tqdm(files[img_set]):
            img_path = i.strip()   
            img        = cv2.imread(img_path)        
            cv2.imwrite(img_path.replace("png", "jpg"), img) # make self jpg

            # mask = os.path.join(annotations_base,
            #                     img_path.split(os.sep)[-2],
            #                     os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png') 
            # data        = cv2.imread(mask, cv2.COLOR_BGR2GRAY)

            mask = os.path.join(annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_color.png').strip()
            data        = cv2.imread(mask, cv2.COLOR_RGB2BGR)
            data = data[:,:,[2,1,0]]
            data = encode_segmap(data)
            new_mask = mask.replace("gtFine_color","gtFine_labelIds")
            cv2.imwrite(new_mask, data)

            unique, counts = np.unique(data, return_counts=True)
            d = dict(zip(unique, counts))             

            for k, v in d.items():
                if k not in total_dict.keys(): total_dict[k] = 0
                total_dict[k] += d[k]

            if((os.path.exists(img_path[:-4] + ".jpg")) and (os.path.exists(new_mask[:-4] + ".png"))):
                ftotal.write(img_path[:-4] + " " + new_mask[:-4] + "\n")    

        ftotal.close()

    w      = open(os.path.join(saveBasePath,'weight.txt'), 'w') 
    od = collections.OrderedDict(sorted(total_dict.items()))
    res = create_class_weight(od)
    [w.write(str(we)+"\n") for we in res.items()]
    w.close()
    print(" weight:", res.values())
    
    print("Generate txt in ImageSets done.")   
   

if __name__ == "__main__":
    data_root = '/home/leyan/DataSet/'
    get_annotation(data_root)   