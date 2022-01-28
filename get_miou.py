import os

from PIL import Image
from tqdm import tqdm

# from deeplabv3_plus.deeplabv3_plus import DeeplabV3 as Model
# from deeplabv3.deeplabv3 import DeepLabv3 as Model
# from pspnet.pspnet import PSPNet as Model
# from unet.unet import Unet as Model
# from segnet.segnet import SegNet as Model
from fcn.fcn import FCN as Model
# from deconvnet.deconvnet import DeconvNet as Model
# from fpn.fpn import FPN as Model

class DataType:
    VOC   = 0
    LANE  = 1
    ICME  = 2  
    COCO  = 3  

class ModelType:
    DEEPLABV3_PLUS   = 0
    DEEPLABV3        = 1
    PSPNET           = 2
    UNET             = 3 
    SEGNET           = 4 
    FCN              = 5
    DeconvNet        = 6
    FPN              = 7

def check_model(o):
    str__ = str(o).split(".")[0].lower()
    if "deeplabv3_plus" in str__: 
        return ModelType.DEEPLABV3_PLUS
    elif "deeplabv3" in str__: 
        return ModelType.DEEPLABV3
    elif "pspnet" in str__: 
        return ModelType.PSPNET
    elif "unet" in str__: 
        return ModelType.UNET
    elif "segnet" in str__: 
        return ModelType.SEGNET  
    elif "fcn" in str__: 
        return ModelType.FCN 
    elif "deconvnet" in str__: 
        return ModelType.DeconvNet 
    elif "fpn" in str__: 
        return ModelType.FPN
    

'''
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''
if __name__ == "__main__":
    #------------------------------#
    root_path = "D://WorkSpace//JupyterWorkSpace"
    dataType = DataType.VOC
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
    #------------------------------#
    #   分类个数+1、如2+1
    #   区分的种类，和json_to_dataset里面的一样
    #   指向VOC数据集所在的文件夹
    #------------------------------#
    if dataType == DataType.VOC:
        num_classes     = 20 + 1
        name_classes    = ["background", "aeroplane", "tvmonitor", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 'diningtable', 
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train"]
        VOCdevkit_path  = os.path.join(root_path, "DataSet/VOCdevkit/VOC2012")

    elif dataType == DataType.LANE:
        num_classes     = 11 + 1
        name_classes    = ["background", "BL", "CL", "DM", "JB", "LA", "PC", "RA", "SA", "SL", "SLA", "SRA"]
        VOCdevkit_path  = os.path.join(root_path, "DataSet/LANEdevkit")

    elif dataType == DataType.ICME:
        num_classes     = 5 + 1
        name_classes    = ["background", "main_lane", "alter_lane", "double_line", "dashed_line", "single_line"]
        VOCdevkit_path  = os.path.join(root_path, "DataSet/ICME2022")
    elif dataType == DataType.COCO:
        num_classes     = 80 + 1
        name_classes    = ["background", "person", "bicycle", "car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant",
                            "stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
                            "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
                            "donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
                            "sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

        VOCdevkit_path  = os.path.join(root_path, "DataSet/ICME2022")
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