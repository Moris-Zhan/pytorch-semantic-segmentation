import sys
sys.path.append(".")

import torch
import torch.nn as nn
import numpy as np
import time, os
import math
import logging
import torch.backends.cudnn as cudnn

# from visdom import Visdom

# import models, datasets, utils
import models

from pynvml import *
from utils.helpers import *
from models.script import get_fit_func
import torch.distributed as dist
from utils.utils_info import write_info


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
        if local_rank == 0:           
            IM_SHAPE = (opt.batch_size, opt.IM_SHAPE[2], opt.IM_SHAPE[0], opt.IM_SHAPE[1])
            rndm_input = torch.autograd.Variable(
                torch.rand(1, opt.IM_SHAPE[2], opt.IM_SHAPE[0], opt.IM_SHAPE[1]), 
                requires_grad = False).cpu()
            opt.writer.add_graph(model, rndm_input)         

            write_info(opt.out_path, model, IM_SHAPE, "model.txt")  
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        if opt.model_path != '':
            #------------------------------------------------------#
            #   權值文件請看README，百度網盤下載
            #------------------------------------------------------#
            print('Load weights {}.'.format(opt.model_path))
            device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict      = model.state_dict()
            pretrained_dict = torch.load(opt.model_path, map_location = device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        # ------------------------------------------------------------------------------
        # model.train()  

        # self.model = model
        model_train = model.train()      
        #-------------------------------------------------------------------#
        #   判断当前batch_size与64的差别，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs         = 64
        opt.Init_lr_fit = max(opt.batch_size / nbs * opt.Init_lr, 1e-4)
        opt.Min_lr_fit  = max(opt.batch_size / nbs * opt.Min_lr, 1e-6)

        self.optimizer = models.get_optimizer(model, opt, opt.optimizer_type)
        self.lr_scheduler_func = get_lr_scheduler(opt.lr_decay_type, opt.Init_lr_fit, opt.Min_lr_fit, opt.UnFreeze_Epoch)

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
        opt.ema = ModelEMA(self.model_train)
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
                self.optimizer = models.get_optimizer(self.model, self.opt, 'adam')                                          
                #-----------------------------------------------------------------------------------------#
                self.loss_history.set_status(freeze=False)
                self.model.unfreeze_backbone()   
                self.loss_history.reset_stop() 
                #-----------------------------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if self.opt.optimizer_type in ['adam', 'adamw'] else 5e-2
                lr_limit_min    = 3e-5 if self.opt.optimizer_type in ['adam', 'adamw'] else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * self.opt.Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * self.opt.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                self.lr_scheduler_func = get_lr_scheduler(self.opt.lr_decay_type, Init_lr_fit, Min_lr_fit, self.opt.UnFreeze_Epoch)
                                                     
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
            
            
            if self.opt.distributed:
                dist.barrier()

        print("End of UnFreeze Training")