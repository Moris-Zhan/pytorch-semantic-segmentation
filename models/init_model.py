import torch
import torch.nn as nn

import importlib
import torch.optim as optim

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_model(opt, pred=False):  
    model = init_dt_model(opt, pred)
    # criterion = init_loss(opt)   
    return model

def init_dt_model(opt, pred=False):
    if opt.net == 'unet':
        from seg_model.unet.nets.unet import Unet
        model = Unet(num_classes=opt.num_classes, pretrained=opt.pretrained, backbone=opt.backbone)

    return model    

# def init_loss(opt):
#     if opt.net == 'unet':
#         from seg_model.unet.nets.unet_training import unet_training
#         criterion       = MultiboxLoss(opt.num_classes, neg_pos_ratio=3.0)
    
#     return criterion

def get_optimizer(model, opt, optimizer_type):    
    optimizer = {
            'adam'  : optim.Adam(model.parameters(), opt.Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), opt.Init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
        }[optimizer_type]   
    return optimizer

def generate_loader(opt):      

    if opt.net == 'unet':
        from seg_model.unet.utils.dataloader import UnetDataset, unet_dataset_collate        
        train_dataset   = UnetDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = UnetDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = unet_dataset_collate  
   

    batch_size      = opt.batch_size
    if opt.distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size // opt.ngpus_per_node
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

    gen             = torch.utils.data.DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val         = torch.utils.data.DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = opt.num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=dataset_collate, sampler=val_sampler) 
    return gen, gen_val