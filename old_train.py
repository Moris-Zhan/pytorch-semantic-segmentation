import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.choose_data import DataType, get_data
from utils.choose_model import ModelType, check_model

from seg_model.deeplabv3_plus.nets.deeplabv3_plus import DeepLab as Model
# from seg_model.deeplabv3.nets.deeplabv3 import DeepLab as Model
# from seg_model.pspnet.nets.pspnet import PSPNet as Model
# from seg_model.unet.nets.unet import Unet  as Model
# from seg_model.segnet.nets.segnet import SegNet as Model
# from seg_model.fcn.nets.fcn import FCN as Model
# from seg_model.deconvnet.nets.deconvnet import DeconvNet as Model
# from seg_model.fpn.nets.fpn import FPN as Model  
        

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
   損失值的具體大小並沒有什麼意義，大和小只在於損失的計算方式，並不是接近於0才好。如果想要讓損失好看點，可以直接到對應的損失函數里面除上10000。
   訓練過程中的損失值會保存在logs文件夾下的loss_%Y_%m_%d_%H_%M_%S文件夾中

4、調參是一門蠻重要的學問，沒有什麼參數是一定好的，現有的參數是我測試過可以正常訓練的參數，因此我會建議用現有的參數。
   但是參數本身並不是絕對的，比如隨著batch的增大學習率也可以增大，效果也會好一些；過深的網絡不要用太大的學習率等等。
   這些都是經驗上，只能靠各位同學多查詢資料和自己試試了。
'''
if __name__ == "__main__":       
    #------------------------------#
    root_path = "D://WorkSpace//JupyterWorkSpace//DataSet"
    VOCdevkit_path, num_classes, cls_weights, _ = get_data(root_path, DataType.LANE)
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
        from seg_model.deeplabv3_plus.nets.deeplabv3_training import weights_init
        from seg_model.deeplabv3_plus.utils.callbacks import LossHistory
        from seg_model.deeplabv3_plus.utils.dataloader import DeeplabDataset, deeplab_dataset_collate
        from seg_model.deeplabv3_plus.utils.utils_fit import fit_one_epoch

        model_path  = "model_data/deeplab_mobilenetv2.pth" # deeplabv3_plus
        input_shape         = [512, 512] 
        backbone    = "mobilenet"

    elif modelType == ModelType.PSPNET:
        from seg_model.pspnet.nets.pspnet_training import weights_init
        from seg_model.pspnet.utils.callbacks import LossHistory
        from seg_model.pspnet.utils.utils_fit import fit_one_epoch
        from seg_model.pspnet.utils.dataloader import PSPnetDataset, pspnet_dataset_collate

        model_path  = "model_data/pspnet_mobilenetv2.pth" # pspnet
        input_shape         = [473, 473] 
        backbone    = "mobilenet"
        #------------------------------------------------------#
        #   是否使用輔助分支
        #   會占用大量顯存
        #------------------------------------------------------#
        aux_branch      = False             # pspnet

    elif modelType == ModelType.UNET:
        from seg_model.unet.nets.unet_training import weights_init
        from seg_model.unet.utils.callbacks import LossHistory
        from seg_model.unet.utils.dataloader import UnetDataset, unet_dataset_collate
        from seg_model.unet.utils.utils_fit import fit_one_epoch

        model_path  = "model_data/unet_vgg_voc.pth" # unet
        input_shape         = [512, 512] 
        backbone    = "vgg"   

    elif modelType == ModelType.SEGNET:
        # from seg_model.segnet.nets.segresnet import SegResNet as Model
        from seg_model.segnet.nets.segnet_training import weights_init
        from seg_model.segnet.utils.callbacks import LossHistory
        from seg_model.segnet.utils.dataloader import SegNetDataset, segnet_dataset_collate
        from seg_model.segnet.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.FCN:
        from seg_model.fcn.nets.fcn_training import weights_init
        from seg_model.fcn.utils.callbacks import LossHistory
        from seg_model.fcn.utils.dataloader import FCNDataset, fcn_dataset_collate
        from seg_model.fcn.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.DeconvNet: 
        from seg_model.deconvnet.nets.deconvnet_training import weights_init
        from seg_model.deconvnet.utils.callbacks import LossHistory
        from seg_model.deconvnet.utils.dataloader import DeconvNetDataset, deconvnet_dataset_collate
        from seg_model.deconvnet.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [224, 224] 

    elif modelType == ModelType.FPN: 
        from seg_model.fpn.nets.fpn_training import weights_init
        from seg_model.fpn.utils.callbacks import LossHistory
        from seg_model.fpn.utils.dataloader import FPNDataset, fpn_dataset_collate
        from seg_model.fpn.utils.utils_fit import fit_one_epoch
        model_path = ""
        input_shape         = [512, 512] 

    elif modelType == ModelType.DEEPLABV3:
        from seg_model.deeplabv3.nets.deeplabv3_training import weights_init
        from seg_model.deeplabv3.utils.callbacks import LossHistory
        from seg_model.deeplabv3.utils.dataloader import DeeplabDataset, deeplab_dataset_collate
        from seg_model.deeplabv3.utils.utils_fit import fit_one_epoch
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
    Freeze_Epoch    = 50
    Freeze_batch_size   = int(8/1)
    Freeze_lr           = 5e-4
    #----------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
    #   占用的顯存較大，網絡所有的參數都會發生改變
    #----------------------------------------------------#
    UnFreeze_Epoch  = 100
    Unfreeze_batch_size = int(4/2)
    Unfreeze_lr         = 5e-5    
    #---------------------------------------------------------------------# 
    #   建議選項：
    #   種類少（幾類）時，設置為True
    #   種類多（十幾類）時，如果batch_size比較大（10以上），那麼設置為True
    #   種類多（十幾類）時，如果batch_size比較小（10以下），那麼設置為False
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
    #   是否提早結束。
    #------------------------------------------------------#
    Early_Stopping  = True
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #------------------------------------------------------#
    num_workers     = 4   
    #----------------------------------------------------------------------------------------------------------------------------#
    if modelType == ModelType.DEEPLABV3_PLUS:
        model   = Model(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)

    elif modelType == ModelType.PSPNET:
        model = Model(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained, aux_branch=aux_branch)
    
    elif modelType == ModelType.UNET:
        model = Model(num_classes=num_classes, pretrained=pretrained, backbone=backbone)

    # [segnet, fcn, deconvnet, fpn, deeplab_v3, segformer]
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
    num_train   = len(train_lines)
    num_val     = len(val_lines)        
    #------------------------------------------------------#
    #   主幹特征提取網絡特征通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Interval_Epoch為凍結訓練的世代
    #   Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False 
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        end_epoch = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch

        if Freeze_Train:
            #------------------------------------#
            #   凍結一定部分訓練
            #------------------------------------#
            loss_history.set_status(freeze=True)
            model.freeze_backbone() 
            loss_history.reset_stop()
            lr          = Freeze_lr
        else:
            #------------------------------------#
            #   解凍後訓練
            #------------------------------------#
            loss_history.set_status(freeze=False)
            model.unfreeze_backbone()   
            loss_history.reset_stop() 
            lr          = Unfreeze_lr
        #-------------------------------------------------------------------#      
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)
        #-------------------------------------------------------------------#

        if modelType in [ModelType.DEEPLABV3_PLUS, ModelType.DEEPLABV3]:
            train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = deeplab_dataset_collate           

        elif modelType == ModelType.PSPNET:
            train_dataset   = PSPnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = PSPnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = pspnet_dataset_collate            
        
        elif modelType == ModelType.UNET:
            train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = unet_dataset_collate           

        elif modelType == ModelType.SEGNET:
            train_dataset   = SegNetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = SegNetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = segnet_dataset_collate
            

        elif modelType == ModelType.FCN:
            train_dataset   = FCNDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = FCNDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = fcn_dataset_collate            
        
        elif modelType == ModelType.DeconvNet:
            train_dataset   = DeconvNetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = DeconvNetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = deconvnet_dataset_collate            

        elif modelType == ModelType.FPN:
            train_dataset   = FPNDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
            val_dataset     = FPNDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
            dataset_collate = fpn_dataset_collate
           
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last=True, collate_fn=dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate)   
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:                            

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")                   
                #-----------------------------------------------------------------------------------------#
                print("End of Freeze Training")
                UnFreeze_flag = True
                #-----------------------------------------------------------------------------------------#
                batch_size = Unfreeze_batch_size   
                end_epoch = UnFreeze_Epoch
                lr          = Unfreeze_lr
               
                optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
                lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)
                
                loss_history.set_status(freeze=False)
                model.unfreeze_backbone()   
                loss_history.reset_stop() 
                #-----------------------------------------------------------------------------------------#    
                        
            # only early stop when UnFreeze Training
            if (UnFreeze_flag and Early_Stopping and loss_history.stopping): break

            if modelType == ModelType.PSPNET:
                # PSPNet
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, aux_branch, num_classes) 
            else:
                # other
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                        epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)           
            lr_scheduler.step()

        print("End of UnFreeze Training")