import os

from PIL import Image
from tqdm import tqdm
from utils.helpers import get_data
    

import argparse, os
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.pspnet_base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')

    parser.add_argument('--gt_dir', type=str, default="test/mask_annotations/")
    parser.add_argument('--classes_path', type=str, default='model_data/coco_classes.txt')
    parser.add_argument("--miou_mode", type=int, default=0 , help="miou mode")  

    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)
    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)


    VOCdevkit_path, num_classes, _, name_classes = get_data(opt.data_root, opt.exp_name)
    #------------------------------#
    compute_mIoU = importlib.import_module("seg_model.%s.utils.utils_metrics"%opt.net).compute_mIoU
    show_results = importlib.import_module("seg_model.%s.utils.utils_metrics"%opt.net).show_results      
    #---------------------------------------------------------------------------#
    miou_mode       = opt.miou_mode
    #-------------------------------------------------------#        
    image_ids       = opt.val_lines
    gt_dir          = os.path.join(VOCdevkit_path, opt.gt_dir)
    miou_out_path   = os.path.join(opt.out_path, "miou_out")
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        model = opt.Model_Pred(num_classes=opt.num_classes)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):   
            image_path = image_id.split()[0] + ".jpg"
            image       = Image.open(image_path)
            image       = model.get_miou_png(image)
            image_id = os.path.basename(image_id)
            image.save(os.path.join(pred_dir, image_id.strip() + ".png"))
            # break
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)