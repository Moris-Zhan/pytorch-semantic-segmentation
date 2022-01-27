import os
import random
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import math
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
VOCdevkit_path  = 'D://WorkSpace/JupyterWorkSpace/DataSet/VOCdevkit'

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
    
    background = [0, 0, 0]
    aeroplane = [128, 0, 0]
    tvmonitor = [0, 128, 0]
    bicycle = [128, 128, 0]
    bird = [0, 0, 128]
    boat = [128, 0, 128]
    bottle = [0, 128, 128]
    bus = [128, 128, 128]
    car = [64, 0, 0]
    cat = [192, 0, 0]
    chair = [64, 128, 0]
    cow = [192, 128, 0]
    diningtable = [64, 0, 128]
    dog = [192, 0, 128]
    horse = [64, 128, 128]
    motorbike = [192, 128, 128]
    person = [0, 64, 0]
    pottedplant = [128, 64, 0]
    sheep = [0, 192, 0]
    sofa = [128, 192, 0]
    train = [0, 64, 128]     

    return np.array([
        background, aeroplane, tvmonitor, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, 
        dog, horse, motorbike, person, pottedplant, sheep, sofa, train])

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

    for year in ["2007", "2012"]:
        segfilepath     = os.path.join(VOCdevkit_path, 'VOC%s/SegmentationClass'%year)
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Segmentation'%year)
        dir_ = os.path.join(VOCdevkit_path, 'VOC%s'%year, "Segmentation")
        if not os.path.exists(dir_): os.makedirs(dir_)
        
        # temp_seg = os.listdir(segfilepath)
        # total_seg = []
        # for seg in temp_seg:
        #     if seg.endswith(".png"):
        #         total_seg.append(seg)
        total_seg = glob(segfilepath + "/*.png")
        # total_dict = {i:1 for i in range(20 + 1)}
        total_dict = {}

        for seg in tqdm(total_seg):
            img        = cv2.imread(seg, cv2.COLOR_BGR2RGB)
            img = img[:,:,[2,1,0]]
            img = encode_segmap(img)
            unique, counts = np.unique(img, return_counts=True)

            d = dict(zip(unique, counts)) 
            for k, v in d.items():
                if k not in total_dict.keys(): total_dict[k] = 0
                total_dict[k] += d[k]
        
        w      = open(os.path.join(dir_,'weight.txt'), 'w') 
        res = create_class_weight(total_dict)
        [w.write(str(we)+"\n") for we in res.items()]

        w.close()
        print(" weight:", res.values())

        num     = len(total_seg)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("traub suze",tr)
        # ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        # ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        # ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        # fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  

        # ftrainval   = open(os.path.join(VOCdevkit_path, 'VOC%s', "Segmentation", 'trainval.txt'), 'w')  
        # ftest       = open(os.path.join(VOCdevkit_path, 'VOC%s', "Segmentation",'test.txt'), 'w')  
        ftrain      = open(os.path.join(VOCdevkit_path, 'VOC%s'%year, "Segmentation",'train.txt'), 'w')  
        fval        = open(os.path.join(VOCdevkit_path, 'VOC%s'%year, "Segmentation",'val.txt'), 'w')  
        
        for i  in list:  
            name=total_seg[i][:-4]+'\n'  
            if i in trainval:  
                # ftrainval.write(name)  
                if i in train:  
                    mask = name.strip()
                    img = name.replace("SegmentationClass", "JPEGImages").strip()
                    ftrain.write(img + " " + mask + "\n")  
                else:  
                    mask = name.strip()
                    img = name.replace("SegmentationClass", "JPEGImages").strip()
                    fval.write(img + " " + mask + "\n")  
            # else:  
            #     ftest.write(name)  
        
        # ftrainval.close()  
        ftrain.close()  
        fval.close()  
        # ftest.close()
        print("Generate txt in ImageSets done.")

    # merge to txt
    for image_set in ["train", "val"]:
        txts = []
        for year in [2007, 2012]:          
            txts.append(os.path.join(VOCdevkit_path,"VOC%s"%(year), "Segmentation" ,"%s.txt"%image_set))

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
        
        if not os.path.exists(os.path.join(VOCdevkit_path, "Segmentation")): os.makedirs(os.path.join(VOCdevkit_path, "Segmentation"))
        with open(os.path.join(VOCdevkit_path, "Segmentation", "%s.txt"%(image_set)), 'w') as fp:
            lines = data.split("\n")
            lines = [fp.write(line + "\n") for line in lines if len(line.split()) > 1]
            print()
            # fp.write(data) 
