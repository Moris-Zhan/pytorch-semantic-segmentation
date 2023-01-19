import sys
sys.path.append(".")

import torch
import torch.nn as nn
import numpy as np
import os
import torch.backends.cudnn as cudnn

import models

from pynvml import *
from utils.helpers import *
from models.script import get_fit_func
import torch.distributed as dist
from utils.utils_info import write_info
from tqdm import tqdm
from PIL import Image
# from .utils_metrics import compute_mIoU
from utils.utils_metrics import compute_mIoU



class Trainer:
    def __init__(self, opt):          
        #------------------------------------------------------#
        #   设置用到的显卡
        #------------------------------------------------------#
        ngpus_per_node  = torch.cuda.device_count()
        if opt.distributed:
            dist.init_process_group(backend="nccl")
            local_rank  = int(os.environ["LOCAL_RANK"])
            rank        = int(os.environ["RANK"])
            device      = torch.device("cuda", local_rank)
            if local_rank == 0:
                print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
                print("Gpu Device Count : ", ngpus_per_node)            
        else:
            device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_rank      = 0
            rank            = 0

        model = models.get_model(opt)  
        # ------------------------------------------------------------------------------- 
        # if local_rank == 0:           
        #     IM_SHAPE = (opt.batch_size, opt.IM_SHAPE[2], opt.IM_SHAPE[0], opt.IM_SHAPE[1])
        #     rndm_input = torch.autograd.Variable(
        #         torch.rand(1, opt.IM_SHAPE[2], opt.IM_SHAPE[0], opt.IM_SHAPE[1]), 
        #         requires_grad = False).cpu()
        #     opt.writer.add_graph(model, rndm_input)         

        #     write_info(opt.out_path, model, IM_SHAPE, "model.txt")  
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        if opt.model_path != '':
            #------------------------------------------------------#
            #   權值文件請看README，百度網盤下載
            #------------------------------------------------------#
            if local_rank == 0: print('Load weights {}.'.format(opt.model_path))
            # device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # model_dict      = model.state_dict()
            # pretrained_dict = torch.load(opt.model_path, map_location = device)
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)
            #------------------------------------------------------#
            #   根据预训练权重的Key和模型的Key进行加载
            #------------------------------------------------------#
            model_dict      = model.state_dict()
            pretrained_dict = torch.load(opt.model_path, map_location = device)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
        # ------------------------------------------------------------------------------
        model_train = model.train()     
        self.optimizer, Init_lr_fit, Min_lr_fit = models.get_optimizer(model, opt, opt.optimizer_type)
        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        self.lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, Init_lr_fit, Min_lr_fit, opt.UnFreeze_Epoch)
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        self.epoch_step      = opt.num_train // opt.batch_size
        self.epoch_step_val  = opt.num_val // opt.batch_size
        
        if self.epoch_step == 0 or self.epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")        

        self.train_loader, self.test_loader = models.generate_loader(opt) 
        
        self.epoch = 0
        self.best_epoch = False
        self.training = False
        self.state = {}

        self.loss_history = LossHistory(opt)
        self.fit_one_epoch = get_fit_func(opt)
        #------------------------------------------------------------------#
        #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
        #   因此torch1.2这里显示"could not be resolve"
        #   torch.cuda.amp: 自動混合精度
        #------------------------------------------------------------------#
        if opt.fp16:
            from torch.cuda.amp import GradScaler as GradScaler
            opt.scaler = GradScaler()
        else:
            opt.scaler = None
            print()
        #----------------------------#
        #   多卡同步Bn
        #----------------------------#
        if opt.sync_bn and ngpus_per_node > 1 and opt.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        elif opt.sync_bn:
            print("Sync_bn is not support in one gpu or not distributed.")
        
        if opt.Cuda:
            if opt.distributed:
                #----------------------------#
                #   多卡平行运行
                #----------------------------#
                model_train = model_train.cuda(local_rank)
                model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
            else:
                model_train = torch.nn.DataParallel(model)
                cudnn.benchmark = True
                model_train = model_train.cuda()

        self.model = model
        self.model_train = model_train        
        #----------------------------#
        #   权值平滑
        #----------------------------#
        # opt.ema = ModelEMA(self.model_train)
        opt.ema = None
        opt.local_rank = local_rank
        self.opt = opt


    def train(self):
        # self.opt.Init_Epoch = 49
        if self.opt.Freeze_Train:
            #------------------------------------#
            #   凍結一定部分訓練
            #------------------------------------#
            self.loss_history.set_status(freeze=True)
            self.model.freeze_backbone() 
            self.loss_history.reset_stop()
        else:
            #------------------------------------#
            #   解凍後訓練
            #------------------------------------#
            self.loss_history.set_status(freeze=False)
            self.model.unfreeze_backbone()   
            self.loss_history.reset_stop() 
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(self.opt.Init_Epoch, self.opt.UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= self.opt.Freeze_Epoch and not self.opt.UnFreeze_flag and self.opt.Freeze_Train:
                #-----------------------------------------------------------------------------------------#
                batch_size = self.opt.Unfreeze_batch_size   
                self.opt.end_epoch = self.opt.UnFreeze_Epoch
                #-----------------------------------------------------------------------------------------#
                self.optimizer, Init_lr_fit, Min_lr_fit = models.get_optimizer(self.model, self.opt, 'adam')
                self.lr_scheduler_func = get_lr_scheduler(self.opt.lr_decay_type, Init_lr_fit, Min_lr_fit, self.opt.UnFreeze_Epoch)
                #-----------------------------------------------------------------------------------------#
                self.loss_history.set_status(freeze=False)
                self.model.unfreeze_backbone()   
                self.loss_history.reset_stop() 

                epoch_step      = self.opt.num_train // batch_size
                epoch_step_val  = self.opt.num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if self.opt.distributed:
                    batch_size = batch_size // self.opt.ngpus_per_node     
                
                self.train_loader, self.test_loader = models.generate_loader(self.opt) 

                self.opt.UnFreeze_flag = True

            # only early stop when UnFreeze Training
            if (self.opt.UnFreeze_flag and self.opt.Early_Stopping and self.loss_history.stopping): break

            set_optimizer_lr(self.optimizer, self.lr_scheduler_func, epoch)

            self.fit_one_epoch(self.model_train, self.model, self.loss_history, self.optimizer, epoch, \
                        self.epoch_step, self.epoch_step_val, self.train_loader, self.test_loader, \
                        self.opt.dice_loss, self.opt.focal_loss, self.opt.cls_weights, self.opt.num_classes, self.opt)           

            if epoch > 0 and epoch % 5 == 0:
                print("Get miou.")
                image_ids       = self.opt.val_lines
                gt_dir          = os.path.join(self.opt.data_path, "test/mask_annotations/")
                miou_out_path   = os.path.join(self.opt.out_path, "miou_out")

                pred_dir    = os.path.join(miou_out_path, 'detection-results')
                os.makedirs(pred_dir, exist_ok=True)
                print("Load model.")
                m = self.opt.Model_Pred()
                print("Load model done.")
                for image_id in tqdm(image_ids):
                    #-------------------------------#
                    #   从文件中读取图像
                    #-------------------------------#
                    image_id = image_id.split(" ")[0]
                    # image_path  = os.path.join(self.dataset_path, "VOC2007/JPEGImages/"+image_id+".jpg")
                    image_path  = os.path.join(image_id+".jpg")
                    image       = Image.open(image_path)
                    #------------------------------#
                    #   获得预测txt
                    #------------------------------#
                    image       = m.get_miou_png(image)
                    image.save(os.path.join(pred_dir, os.path.basename(image_id) + ".png"))
                    
                print("Calculate miou.")            
                compute_mIoU(gt_dir, pred_dir, image_ids, self.opt.num_classes, None)  # 执行计算mIoU的函数
                
            if self.opt.distributed:
                dist.barrier()

        print("End of UnFreeze Training")