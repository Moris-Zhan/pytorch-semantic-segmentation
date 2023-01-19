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
    return model

# [segnet, fcn, deconvnet, fpn, deeplab_v3, segformer]
def init_dt_model(opt, pred=False):
    if opt.net == 'unet':
        from seg_model.unet.nets.unet import Unet
        model = Unet(num_classes=opt.num_classes, pretrained=opt.pretrained, backbone=opt.backbone)
    elif opt.net == 'pspnet':
        from seg_model.pspnet.nets.pspnet import PSPNet
        model = PSPNet(num_classes=opt.num_classes, backbone=opt.backbone, downsample_factor=opt.downsample_factor, pretrained=opt.pretrained, aux_branch=opt.aux_branch)
    elif opt.net == 'deeplab_v3':
        from seg_model.deeplabv3.nets.deeplabv3 import DeepLab
        model = DeepLab(num_classes=opt.num_classes, pretrained=opt.pretrained)
    elif opt.net == 'deeplab_v3_plus':
        from seg_model.deeplabv3_plus.nets.deeplabv3_plus import DeepLab
        model = DeepLab(num_classes=opt.num_classes, backbone=opt.backbone, pretrained=opt.pretrained, downsample_factor=opt.downsample_factor)
    elif opt.net == 'segnet':
        from seg_model.segnet.nets.segnet import SegNet
        model = SegNet(num_classes=opt.num_classes, pretrained=opt.pretrained)
    elif opt.net == 'fcn':
        from seg_model.fcn.nets.fcn import FCN
        model = FCN(num_classes=opt.num_classes, pretrained=opt.pretrained)
    elif opt.net == 'deconvnet':
        from seg_model.deconvnet.nets.deconvnet import DeconvNet
        model = DeconvNet(num_classes=opt.num_classes, pretrained=opt.pretrained)
    elif opt.net == 'fpn':
        from seg_model.fpn.nets.fpn import FPN 
        model = FPN(num_classes=opt.num_classes, pretrained=opt.pretrained)
    elif opt.net == 'segformer':
        from seg_model.segformer.nets.segformer import SegFormer 
        model = SegFormer(num_classes=opt.num_classes, phi=opt.phi,pretrained=opt.pretrained)

    return model    


def get_optimizer(model, opt, optimizer_type):    
    if opt.net == 'deeplab_v3_plus':
        from configs.deeplab_v3_plus_base import reset_lr
    elif opt.net == 'unet':
        from configs.unet_base import reset_lr
    elif opt.net == 'pspnet':
        from configs.pspnet_base import reset_lr
    else:
        def reset_lr(Init_lr, Min_lr, optimizer_type, backbone, batch_size):
            #-----------------------------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            #-------------------------------------------------------------------#                
            nbs             = 16
            lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
            if backbone == "xception":
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            return Init_lr_fit, Min_lr_fit
            
    Init_lr_fit, Min_lr_fit = reset_lr(opt.Init_lr, opt.Min_lr, optimizer_type, opt.backbone, opt.batch_size)

    optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'adamw' : optim.AdamW(model.parameters(), Init_lr_fit, betas = (opt.momentum, 0.999), weight_decay = opt.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = opt.momentum, nesterov=True, weight_decay = opt.weight_decay)
        }[optimizer_type]   
    return optimizer, Init_lr_fit, Min_lr_fit

def generate_loader(opt):      

    if opt.net == 'unet':
        from seg_model.unet.utils.dataloader import UnetDataset, unet_dataset_collate        
        train_dataset   = UnetDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = UnetDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = unet_dataset_collate 
    elif opt.net == 'pspnet':
        from seg_model.pspnet.utils.dataloader import PSPnetDataset, pspnet_dataset_collate
        train_dataset   = PSPnetDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = PSPnetDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = pspnet_dataset_collate
    elif opt.net == 'deeplab_v3':
        from seg_model.deeplabv3.utils.dataloader import DeeplabDataset, deeplab_dataset_collate
        train_dataset   = DeeplabDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = DeeplabDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = deeplab_dataset_collate
    elif opt.net == 'deeplab_v3_plus':
        from seg_model.deeplabv3_plus.utils.dataloader import DeeplabDataset, deeplab_dataset_collate
        train_dataset   = DeeplabDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = DeeplabDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = deeplab_dataset_collate
    elif opt.net == 'segnet':
        from seg_model.segnet.utils.dataloader import SegNetDataset, segnet_dataset_collate
        train_dataset   = SegNetDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = SegNetDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = segnet_dataset_collate
    elif opt.net == 'fcn':
        from seg_model.fcn.utils.dataloader import FCNDataset, fcn_dataset_collate
        train_dataset   = FCNDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = FCNDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = fcn_dataset_collate   
    elif opt.net == 'deconvnet':
        from seg_model.deconvnet.utils.dataloader import DeconvNetDataset, deconvnet_dataset_collate
        train_dataset   = DeconvNetDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = DeconvNetDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = deconvnet_dataset_collate
    elif opt.net == 'fpn':
        from seg_model.fpn.utils.dataloader import FPNDataset, fpn_dataset_collate
        train_dataset   = FPNDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = FPNDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = fpn_dataset_collate
    elif opt.net == 'segformer':
        from seg_model.segformer.utils.dataloader import SegmentationDataset, seg_dataset_collate
        train_dataset   = SegmentationDataset(opt.train_lines, opt.input_shape, opt.num_classes, True, opt.data_path)
        val_dataset     = SegmentationDataset(opt.val_lines, opt.input_shape, opt.num_classes, False, opt.data_path)
        dataset_collate = seg_dataset_collate

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