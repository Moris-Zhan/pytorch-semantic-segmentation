import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

# from deeplabv3_plus.nets.deeplabv3_plus import DeepLab as Model
# from deeplabv3.nets.deeplabv3 import DeepLab as Model
# from pspnet.nets.pspnet import PSPNet as Model
# from unet.nets.unet import Unet  as Model
# from segnet.nets.segnet import SegNet as Model
from fcn.nets.fcn import FCN as Model
# from deconvnet.nets.deconvnet import DeconvNet as Model
# from fpn.nets.fpn import FPN as Model



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
    

def get_cls_weight(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [float(line.split(", ")[1].split(")")[0]) for line in lines]
        cls_weights     = np.array(lines, np.float32)
    return cls_weights
        

'''
訓練自己的語義分割模型一定需要注意以下幾點：
1、訓練前仔細檢查自己的格式是否滿足要求，該庫要求數據集格式為VOC格式，需要準備好的內容有輸入圖片和標簽
   輸入圖片為.jpg圖片，無需固定大小，傳入訓練前會自動進行resize。
   灰度圖會自動轉成RGB圖片進行訓練，無需自己修改。
   輸入圖片如果後綴非jpg，需要自己批量轉成jpg後再開始訓練。

   標簽為png圖片，無需固定大小，傳入訓練前會自動進行resize。
   由於許多同學的數據集是網絡上下載的，標簽格式並不符合，需要再度處理。一定要注意！標簽的每個像素點的值就是這個像素點所屬的種類。
   網上常見的數據集總共對輸入圖片分兩類，背景的像素點值為0，目標的像素點值為255。這樣的數據集可以正常運行但是預測是沒有效果的！
   需要改成，背景的像素點值為0，目標的像素點值為1。

2、訓練好的權值文件保存在logs文件夾中，每個epoch都會保存一次，如果只是訓練了幾個step是不會保存的，epoch和step的概念要捋清楚一下。
   在訓練過程中，該代碼並沒有設定只保存最低損失的，因此按默認參數訓練完會有100個權值，如果空間不夠可以自行刪除。
   這個並不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一點，為了滿足大多數的需求，還是都保存可選擇性高。

3、損失值的大小用於判斷是否收斂，比較重要的是有收斂的趨勢，即驗證集損失不斷下降，如果驗證集損失基本上不改變的話，模型基本上就收斂了。
   損失值的具體大小並沒有什麽意義，大和小只在於損失的計算方式，並不是接近於0才好。如果想要讓損失好看點，可以直接到對應的損失函數里面除上10000。
   訓練過程中的損失值會保存在logs文件夾下的loss_%Y_%m_%d_%H_%M_%S文件夾中

4、調參是一門蠻重要的學問，沒有什麽參數是一定好的，現有的參數是我測試過可以正常訓練的參數，因此我會建議用現有的參數。
   但是參數本身並不是絕對的，比如隨著batch的增大學習率也可以增大，效果也會好一些；過深的網絡不要用太大的學習率等等。
   這些都是經驗上，只能靠各位同學多查詢資料和自己試試了。
'''
if __name__ == "__main__":       
    #------------------------------#
    dataType = DataType.VOC
    #------------------------------#
    root_path = "D://WorkSpace//JupyterWorkSpace//DataSet"
    #-------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #-------------------------------#
    Cuda = True    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主幹網絡的預訓練權重，此處使用的是主幹的權重，因此是在模型構建的時候進行加載的。
    #   如果設置了model_path，則主幹的權值無需加載，pretrained的值無意義。
    #   如果不設置model_path，pretrained = True，此時僅加載主幹開始訓練。
    #   如果不設置model_path，pretrained = False，Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #--------------------------------------------------------------------------------------------------------------------------
    # pretrained  = False
    pretrained  = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   權值文件的下載請看README，可以通過網盤下載。模型的 預訓練權重 對不同數據集是通用的，因為特征是通用的。
    #   模型的 預訓練權重 比較重要的部分是 主幹特征提取網絡的權值部分，用於進行特征提取。
    #   預訓練權重對於99%的情況都必須要用，不用的話主幹部分的權值太過隨機，特征提取效果不明顯，網絡訓練的結果也不會好
    #   訓練自己的數據集時提示維度不匹配正常，預測的東西都不一樣了自然維度不匹配
    #
    #   如果訓練過程中存在中斷訓練的操作，可以將model_path設置成logs文件夾下的權值文件，將已經訓練了一部分的權值再次載入。
    #   同時修改下方的 凍結階段 或者 解凍階段 的參數，來保證模型epoch的連續性。
    #   
    #   當model_path = ''的時候不加載整個模型的權值。
    #
    #   此處使用的是整個模型的權重，因此是在train.py進行加載的，pretrain不影響此處的權值加載。
    #   如果想要讓模型從主幹的預訓練權值開始訓練，則設置model_path = ''，pretrain = True，此時僅加載主幹。
    #   如果想要讓模型從0開始訓練，則設置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #   一般來講，從0開始訓練效果會很差，因為權值太過隨機，特征提取效果不明顯。
    #
    #   網絡一般不從0開始訓練，至少會使用主幹部分的權值，有些論文提到可以不用預訓練，主要原因是他們 數據集較大 且 調參能力優秀。
    #   如果一定要訓練網絡的主幹部分，可以了解imagenet數據集，首先訓練分類模型，分類模型的 主幹部分 和該模型通用，基於此進行訓練。

    #   輸入圖片的大小

    #   所使用的的主幹網絡：
    #   mobilenet、xception 
    #   
    #   在使用xception作為主幹網絡時，建議在訓練參數設置部分調小學習率，如：
    #   Freeze_lr   = 3e-4
    #   Unfreeze_lr = 1e-5    
    #----------------------------------------------------------------------------------------------------------------------------#
    modelType = check_model(Model.__module__)
    if modelType == ModelType.DEEPLABV3_PLUS:
        from deeplabv3_plus.nets.deeplabv3_training import weights_init
        from deeplabv3_plus.utils.callbacks import LossHistory
        from deeplabv3_plus.utils.dataloader import DeeplabDataset, deeplab_dataset_collate
        from deeplabv3_plus.utils.utils_fit import fit_one_epoch

        model_path  = "deeplabv3_plus/weight/deeplab_mobilenetv2.pth" # deeplabv3_plus
        input_shape         = [512, 512] 
        backbone    = "mobilenet"

    elif modelType == ModelType.PSPNET:
        from pspnet.nets.pspnet_training import weights_init
        from pspnet.utils.callbacks import LossHistory
        from pspnet.utils.utils_fit import fit_one_epoch
        from pspnet.utils.dataloader import PSPnetDataset, pspnet_dataset_collate

        model_path  = "pspnet/weight/pspnet_mobilenetv2.pth" # pspnet
        input_shape         = [473, 473] 
        backbone    = "mobilenet"
        #------------------------------------------------------#
        #   是否使用辅助分支
        #   会占用大量显存
        #------------------------------------------------------#
        aux_branch      = False             # pspnet

    elif modelType == ModelType.UNET:
        from unet.nets.unet_training import weights_init
        from unet.utils.callbacks import LossHistory
        from unet.utils.dataloader import UnetDataset, unet_dataset_collate
        from unet.utils.utils_fit import fit_one_epoch

        model_path  = "unet/weight/unet_vgg_voc.pth" # unet
        input_shape         = [512, 512] 
        backbone    = "vgg"   

    elif modelType == ModelType.SEGNET:
        # from segnet.nets.segresnet import SegResNet as Model
        from segnet.nets.segnet_training import weights_init
        from segnet.utils.callbacks import LossHistory
        from segnet.utils.dataloader import SegNetDataset, segnet_dataset_collate
        from segnet.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.FCN:
        from fcn.nets.fcn_training import weights_init
        from fcn.utils.callbacks import LossHistory
        from fcn.utils.dataloader import FCNDataset, fcn_dataset_collate
        from fcn.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.DeconvNet: 
        from deconvnet.nets.deconvnet_training import weights_init
        from deconvnet.utils.callbacks import LossHistory
        from deconvnet.utils.dataloader import DeconvNetDataset, deconvnet_dataset_collate
        from deconvnet.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.FPN: 
        from fpn.nets.fpn_training import weights_init
        from fpn.utils.callbacks import LossHistory
        from fpn.utils.dataloader import FPNDataset, fpn_dataset_collate
        from fpn.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.DEEPLABV3:
        from deeplabv3.nets.deeplabv3_training import weights_init
        from deeplabv3.utils.callbacks import LossHistory
        from deeplabv3.utils.dataloader import DeeplabDataset, deeplab_dataset_collate
        from deeplabv3.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 
    #---------------------------------------------------------#
    #   下采樣的倍數8、16 
    #   8下采樣的倍數較小、理論上效果更好，但也要求更大的顯存
    #---------------------------------------------------------#
    downsample_factor   = 16   
    #----------------------------------------------------#
    #   訓練分為兩個階段，分別是凍結階段和解凍階段。
    #   顯存不足與數據集大小無關，提示顯存不足請調小batch_size。
    #   受到BatchNorm層影響，batch_size最小為2，不能為1。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特征提取網絡不發生改變
    #   占用的顯存較小，僅對網絡進行微調
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = int(8/4)
    Freeze_lr           = 5e-4
    #----------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
    #   占用的顯存較大，網絡所有的參數都會發生改變
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = int(4/2)
    Unfreeze_lr         = 5e-5    
    #---------------------------------------------------------------------# 
    #   建議選項：
    #   種類少（幾類）時，設置為True
    #   種類多（十幾類）時，如果batch_size比較大（10以上），那麽設置為True
    #   種類多（十幾類）時，如果batch_size比較小（10以下），那麽設置為False
    #---------------------------------------------------------------------# 
    dice_loss       = False
    #---------------------------------------------------------------------# 
    #   是否使用focal loss來防止正負樣本不平衡
    #---------------------------------------------------------------------# 
    focal_loss      = True
    #------------------------------------------------------#
    #   是否進行凍結訓練，默認先凍結主幹訓練後解凍訓練。
    #------------------------------------------------------#
    Freeze_Train    = True
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #------------------------------------------------------#
    num_workers     = 4
    #-------------------------------#    
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
    elif dataType == DataType.LANE:
        #   LANEdevkit
        VOCdevkit_path  = os.path.join(root_path, "LANEdevkit")    
        num_classes = 11 + 1
        cls_weights     = get_cls_weight("%s/Segmentation/weight_train.txt" %(VOCdevkit_path))
    elif dataType == DataType.ICME:
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "ICME2022")    
        num_classes = 5 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/weight.txt" %(VOCdevkit_path))  
    elif dataType == DataType.COCO:
        #   ICME2022
        VOCdevkit_path  = os.path.join(root_path, "COCO")    
        num_classes = 80 + 1   
        cls_weights     = get_cls_weight("%s/Segmentation/2014_weight.txt" %(VOCdevkit_path))
    #----------------------------------------------------------------------------------------------------------------------------#
    if modelType == ModelType.DEEPLABV3_PLUS:
        model   = Model(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)

    elif modelType == ModelType.PSPNET:
        model = Model(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained, aux_branch=aux_branch)
    
    elif modelType == ModelType.UNET:
        model = Model(num_classes=num_classes, pretrained=pretrained, backbone=backbone)

    elif modelType in [ModelType.SEGNET, ModelType.FCN, ModelType.DeconvNet, ModelType.FPN, ModelType.DEEPLABV3]:
        model = Model(num_classes=num_classes, pretrained=pretrained)  
    

    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   權值文件請看README，百度網盤下載
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history = LossHistory(model_train)    
    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "Segmentation//train.txt"),"r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "Segmentation//val.txt"),"r") as f:
        val_lines = f.readlines()
        
    #------------------------------------------------------#
    #   主幹特征提取網絡特征通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Interval_Epoch為凍結訓練的世代
    #   Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    #------------------------------------------------------#
    if False:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("數據集過小，無法進行訓練，請擴充數據集。")

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.DEEPLABV3]:
            train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = deeplab_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last = True, collate_fn = deeplab_dataset_collate)

        elif modelType == ModelType.PSPNET:
            train_dataset   = PSPnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = PSPnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = pspnet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last = True, collate_fn = pspnet_dataset_collate)
        
        elif modelType == ModelType.UNET:
            train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = unet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate)

        elif modelType == ModelType.SEGNET:
            train_dataset   = SegNetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = SegNetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = segnet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = segnet_dataset_collate)

        elif modelType == ModelType.FCN:
            train_dataset   = FCNDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = FCNDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = fcn_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = fcn_dataset_collate)

        elif modelType == ModelType.DeconvNet:
            train_dataset   = DeconvNetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = DeconvNetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = deconvnet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deconvnet_dataset_collate)

        elif modelType == ModelType.FPN:
            train_dataset   = FPNDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = FPNDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = fpn_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = fpn_dataset_collate)
    
        #------------------------------------#
        #   凍結一定部分訓練
        #------------------------------------#
        if Freeze_Train:
            loss_history.set_status(freeze=True)

            if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.PSPNET]:
                # Deeplab / PSPNet
                for param in model.backbone.parameters():
                    param.requires_grad = False
            elif modelType in[ModelType.UNET,  ModelType.SEGNET, ModelType.FCN, ModelType.DeconvNet, ModelType.FPN, ModelType.DEEPLABV3]:
                # Unet / SegNet / FCN / DeconvNet
                model.freeze_backbone()   

            
        for epoch in range(start_epoch, end_epoch):
            if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.UNET, ModelType.SEGNET, ModelType.FCN, ModelType.DeconvNet, ModelType.FPN, ModelType.DEEPLABV3]:
                # Deeplab / Unet / SegNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)
            
            elif modelType == ModelType.PSPNET:
                # PSPNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, aux_branch, num_classes)             
                          
            lr_scheduler.step()
    
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        epoch_step      = len(train_lines) // batch_size
        epoch_step_val  = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("數據集過小，無法進行訓練，請擴充數據集。")

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.DEEPLABV3]:
            train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = deeplab_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last = True, collate_fn = deeplab_dataset_collate)

        elif modelType == ModelType.PSPNET:
            train_dataset   = PSPnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = PSPnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = pspnet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last = True, collate_fn = pspnet_dataset_collate)
        
        elif modelType == ModelType.UNET:
            train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = unet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate)

        elif modelType == ModelType.SEGNET:
            train_dataset   = SegNetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = SegNetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = segnet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = segnet_dataset_collate)

        elif modelType == ModelType.FCN:
            train_dataset   = FCNDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = FCNDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = fcn_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = fcn_dataset_collate)
        
        elif modelType == ModelType.DeconvNet:
            train_dataset   = DeconvNetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = DeconvNetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = deconvnet_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = deconvnet_dataset_collate)

        elif modelType == ModelType.FPN:
            train_dataset   = FPNDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = FPNDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = fpn_dataset_collate)
            gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = fpn_dataset_collate)    

            
    if Freeze_Train:
        loss_history.set_status(freeze=False)

        if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.PSPNET]:
            # Deeplab / PSPNet
            for param in model.backbone.parameters():
                param.requires_grad = True
        elif modelType in[ModelType.UNET,  ModelType.SEGNET, ModelType.FCN, ModelType.DeconvNet, ModelType.FPN, ModelType.DEEPLABV3]:
            # Unet / SegNet
            model.unfreeze_backbone() 

        for epoch in range(start_epoch,end_epoch):
            if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.UNET, ModelType.SEGNET, ModelType.FCN, ModelType.DeconvNet, ModelType.FPN, ModelType.DEEPLABV3]:
                # Deeplab / Unet / SegNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)
            
            elif modelType == ModelType.PSPNET:
                # PSPNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, aux_branch, num_classes)        
                              
            lr_scheduler.step()
