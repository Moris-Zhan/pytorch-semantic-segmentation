import os
import numpy as np
import argparse

import torch
from torchvision.utils import save_image
# from data.CamVid_loader import CamVidDataset
from dataset.utils import decode_segmap, decode_seg_map_sequence
from dataset import Cityscapes, Coco, Pascal

from utils.metrics import Evaluator

from model.FPN import FPN
from model.UNet import UNet
from model.SegNet import SegNet
from model.FCN import FCNs
from model.DeConvNet import DeConvNet
from model.PSPNet import PSPNet
from model.DeepLabV3 import DeepLabV3
from model.DeepLabv3_plus import DeepLabv3_plus
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from glob import glob

if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', default='Cityscapes', type=str, help='training dataset, Cityscapes, Coco, Pascal')
    # parser.add_argument('--net', type=str, default="FPN", help='FCN/SegNet/DeconvNet/UNet/PSPNet/DeepLabV3/DeepLabV3+/FPN') 
    parser.add_argument('--model', type=str, default="PSPNet", help='FPN/FCN/DeConvNet/UNet/SegNet/PSPNet/DeepLabV3/DeepLabv3_plus/')
    parser.add_argument('--start_epoch', help='starting epoch', default=1, type=int)
    parser.add_argument('--epochs', help='number of iterations to train', default=2000, type=int)
    parser.add_argument('--save_dir', help='directory to save models', default="D:\\disk\\midterm\\experiment\\code\\semantic\\FPN\\FPN\\run", type=str)
    parser.add_argument('--num_workers', help='number of worker to load data', default=0, type=int)
    # cuda
    parser.add_argument('--cuda', help='whether use multiple GPUs', default=True, action='store_true')
    # batch size
    parser.add_argument('--batch_size', help='batch_size', default=2, type=int)

    # config optimization
    parser.add_argument('--o', help='training optimizer', default='sgd', type=str)
    parser.add_argument('--lr', help='starting learning rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', help='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', help='step to do learning rate decay, uint is epoch', default=500, type=int)
    parser.add_argument('--lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)

    # set training session
    parser.add_argument('--session', help='training session', default=1, type=int)

    # resume trained model
    parser.add_argument('--resume', help='resume checkpoint or not', default=False, type=bool)
    parser.add_argument('--checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', help='checkpoint to load model', default=0, type=int)

    # log and display
    parser.add_argument('--use_tfboard', help='whether use tensorflow tensorboard', default=True, type=bool)

    # configure validation
    parser.add_argument('--no_val', help='not do validation', default=False, type=bool)
    parser.add_argument('--eval_interval', help='iterval to do evaluate', default=2, type=int)

    # parser.add_argument('--base-size', type=int, default=512, help='base image size')
    # parser.add_argument('--crop-size', type=int, default=512, help='crop image size')
    

    # parser.add_argument('--batch_size', help='batch_size', default=1, type=int)
    # parser.add_argument('--base_size', type=int, default=1024, help='base image size')
    # parser.add_argument('--crop_size', type=int, default=512, help='crop image size')

    parser.add_argument('--base_size', type=int, default=1024, help='base image size')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')

    # test confit
    parser.add_argument('--plot', help='wether plot test result image', default=False, type=bool)
    parser.add_argument('--experiment_dir', help='dir of experiment', type=str, default = "run\Cityscapes\PSPNet\experiment_10")

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    train_set, val_set, test_set = None, None, None
    # Define Dataloader
    if opt.dataset == 'Cityscapes':
        val_set = Cityscapes.CityscapesSegmentation(opt, split='val')        
    elif opt.dataset == 'Coco':
        val_set = Coco.COCOSegmentation(opt, split='val')
    elif opt.dataset == 'Pascal':
        val_set = Pascal.VOCSegmentation(opt, split='val')
        # Pascal.VOCSegmentation(opt, split='val')

    num_class = val_set.NUM_CLASSES
    save_image_path = './test/{}/'.format(opt.dataset)
    if not os.path.exists(save_image_path): os.makedirs(save_image_path)
    kwargs = {'num_workers': opt.num_workers, 'pin_memory': True}
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, drop_last=True, **kwargs)


    # Define network
    model = None
    if opt.model == 'FPN':
        model = FPN(num_class, back_bone_name= 'resnet101')
        opt.checkname = 'FPN-' + model.back_bone_name
    elif opt.model == 'UNet':
        model = UNet(n_channels=3, n_classes=num_class, bilinear=True)
        opt.checkname = opt.model
    elif opt.model == "FCN":
        model = FCNs(n_class=num_class)
        opt.checkname = opt.model
    elif opt.model == "DeConvNet":
        model = DeConvNet(n_class=num_class)
        opt.checkname = opt.model
    elif opt.model == 'SegNet':
        model = SegNet(num_classes = num_class)
        opt.checkname = opt.model
    elif opt.model == 'PSPNet':
        # batchnorm require batch >=2
        model = PSPNet(n_classes = num_class, PSP_h = opt.base_size, PSP_w=opt.crop_size, Auxiliray=True)
        opt.checkname = opt.model
        opt.use_auxiliary = model.Auxiliray
    elif opt.model == 'DeepLabV3':
        model = DeepLabV3(num_classes = num_class)
        opt.checkname = opt.model
    elif opt.model == 'DeepLabv3_plus':
        model = DeepLabv3_plus(nInputChannels=3, n_classes=num_class, os=16, pretrained=True)  
        opt.checkname = opt.model

    evaluator = Evaluator(num_class)


    # Trained model path and name
    experiment_dir = opt.experiment_dir
    model_name = glob(os.path.join(opt.experiment_dir, "*.pkl"))[0]
    load_name = os.path.join(experiment_dir, 'checkpoint.pth.tar')

    # Load save/trained model
    if not os.path.isfile(model_name):
        raise RuntimeError("=> no model found at '{}'".format(model_name))
    print('====>loading trained model from ' + model_name)
    if not os.path.isfile(load_name):
        raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))
    print('====>loading trained model from ' + load_name)

    
    if opt.model != 'PSPNet':
        model = torch.load(model_name)
    checkpoint = torch.load(load_name)

    if opt.cuda:
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
    else:
        model.load_state_dict(checkpoint['state_dict'])


    # test
    Acc = []
    Acc_class = []
    mIoU = []
    FWIoU = []
    results = []
    orign_img = []        

    with tqdm(total=len(val_loader)) as pbar:
        for iter, batch in enumerate(val_loader):
            imgs, targets = batch['image'], batch['label']        
            imgs = Variable(imgs.to(device=device))
            targets = Variable(targets.to(device=device))
            if opt.model == 'PSPNet':
                imgs = PSPNet.permute(imgs)
                targets = PSPNet.permute(targets)

            with torch.no_grad():
                if opt.model == 'PSPNet':
                    output, aux_output = model(imgs)
                else:
                    output = model(imgs)

            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            targets = targets.cpu().numpy()
            evaluator.add_batch(targets, pred)

            # show result
            pred_rgb = decode_seg_map_sequence(pred, opt.dataset)
            orign_img.extend(imgs.cpu())
            results.extend(pred_rgb)  
            pbar.set_description('Pred Batch Images : %s' % ( iter * opt.batch_size))
            pbar.update(1)
            break

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    ratio = 0.3

    # save Img
    with tqdm(total= len(orign_img)) as pbar:
        for idx, (img, pred) in enumerate(zip(orign_img, results)):
            # merge = torch.cat([img.unsqueeze(0), pred.unsqueeze(0).type(torch.FloatTensor)], dim = 0)
            img = img.mul_(ratio)
            pred = pred.mul_(1 - ratio)
            merge = img.add(pred.type(torch.FloatTensor))  
            if opt.model == 'PSPNet':
                merge = PSPNet.permute(merge)

            save_image(merge, save_image_path + 'val{}_jpg.png'.format(idx))
            pbar.set_description('Save Pred Images : %s' % ( idx ))
            pbar.update(1)

    print('Mean evaluate result on dataset {}'.format(opt.dataset))
    print('Acc:{:.3f}\tAcc_class:{:.3f}\nmIoU:{:.3f}\tFWIoU:{:.3f}'.format(Acc, Acc_class, mIoU, FWIoU))


