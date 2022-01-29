import os

from PIL import Image
from tqdm import tqdm
from helps.choose_data import DataType, get_data
from helps.choose_model import ModelType, check_model

# from deeplabv3_plus.deeplabv3_plus import DeeplabV3 as Model
# from deeplabv3.deeplabv3 import DeepLabv3 as Model
# from pspnet.pspnet import PSPNet as Model
# from unet.unet import Unet as Model
from segnet.segnet import SegNet as Model
# from fcn.fcn import FCN as Model
# from deconvnet.deconvnet import DeconvNet as Model
# from fpn.fpn import FPN as Model    

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #------------------------------#
    #   分类个数+1、如2+1
    #   区分的种类，和json_to_dataset里面的一样
    #   指向VOC数据集所在的文件夹
    root_path = "D://WorkSpace//JupyterWorkSpace//DataSet"
    VOCdevkit_path, num_classes, _, name_classes = get_data(root_path, DataType.VOC)
    #------------------------------#
    modelType = check_model(Model.__module__)
    if modelType == ModelType.DEEPLABV3_PLUS: 
        from deeplabv3_plus.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.DEEPLABV3: 
        from deeplabv3.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.PSPNET: 
        from pspnet.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.UNET: 
        from unet.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.SEGNET: 
        from segnet.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.FCN: 
        from fcn.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.DeconvNet: 
        from deconvnet.utils.utils_metrics import compute_mIoU, show_results

    elif modelType == ModelType.FPN: 
        from fpn.utils.utils_metrics import compute_mIoU, show_results
  
    
    #---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #-------------------------------------------------------#        
    image_ids       = open(os.path.join(VOCdevkit_path, "Segmentation/val.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "test/mask_annotations/")
    miou_out_path   = os.path.join("miou_out", Model.__module__)
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        model = Model()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):   
            image_path = image_id.split()[0] + ".jpg"
            image       = Image.open(image_path)
            image       = model.get_miou_png(image)
            image_id = os.path.basename(image_id)
            image.save(os.path.join(pred_dir, image_id + ".png"))
            # break
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)