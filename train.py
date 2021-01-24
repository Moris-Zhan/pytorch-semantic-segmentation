import numpy as np
import argparse
import torch
from torch.autograd import Variable

from dataset import Cityscapes, Coco, Pascal
from torch.utils.data import DataLoader
from model.FPN import FPN
from model.UNet import UNet
from model.SegNet import SegNet
from model.FCN import FCNs
from model.DeConvNet import DeConvNet
from model.PSPNet import PSPNet
from model.DeepLabV3 import DeepLabV3
from model.DeepLabv3_plus import DeepLabv3_plus
from losses import SegmentationLosses
from tqdm import tqdm

from utils.saver import Saver
from utils.metrics import Evaluator
from utils.pytorchtools import EarlyStopping
import statistics

def eval_model(model, val_loader):
    model.eval()
    evaluator = Evaluator(num_class)
    evaluator.reset()
    test_loss = 0.0
    with tqdm(total=len(val_loader)) as pbar:
        for iter, batch in enumerate(val_loader):
            imgs, targets = batch['image'], batch['label']        
            imgs = Variable(imgs.to(device=device))
            targets = Variable(targets.to(device=device))
            if opt.model == 'PSPNet':
                imgs = PSPNet.permute(imgs)
                targets = PSPNet.permute(targets)

            if opt.use_auxiliary:
                # PSPNet
                r = 0.4
                with torch.no_grad():
                    outputs, aux_outputs = model(imgs) # masks_pred
                loss = criterion(outputs, targets.long())
                loss_aux = criterion(aux_outputs, targets.long())
                loss = loss * (1-r) + loss_aux * r
            else:
                with torch.no_grad():
                    outputs = model(imgs) # masks_pred
                loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            # pbar.set_description('Test img shape=%s Loss:%.3f' % (str(imgs.shape), (test_loss/(iter+1))))
            pbar.set_description("Validation: ")
            pred = outputs.data.cpu().numpy()
            targets = targets.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(targets, pred)
            pbar.update(1)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    return Acc, Acc_class, mIoU, FWIoU

if __name__ == '__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', default='Pascal', type=str, help='training dataset, Cityscapes, Coco, Pascal')
    parser.add_argument('--model', type=str, default="PSPNet", help='FPN/FCN/DeConvNet/UNet/SegNet/PSPNet/DeepLabV3/DeepLabv3_plus/') 
    parser.add_argument('--start_epoch', default=1, type=int, help='starting epoch')
    parser.add_argument('--epochs', default=5
    , type=int, help='number of iterations to train' )
    parser.add_argument('--save_dir', default=None, nargs=argparse.REMAINDER, help='directory to save models' )
    parser.add_argument('--num_workers', default=0, type=int, help='number of worker to load data' )

    # cuda
    parser.add_argument('--cuda', default=True, type=bool, help='whether use CUDA')
    # multiple GPUs
    parser.add_argument('--mGPUs', default=False, help='whether use multiple GPUs')
    parser.add_argument('--gpu_ids', default='0', type=str, help='use which gpu to train, must be a comma-separated list of integers only (defalt=0)' )
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of cpu threads to use during batch generation")

    # batch size
    # parser.add_argument('--batch_size', help='batch_size', default=2, type=int)

    # config optimization
    parser.add_argument('--optimizer', help='training optimizer', default='sgd', type=str)
    parser.add_argument('--lr', help='starting learning rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', help='weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', help='step to do learning rate decay, uint is epoch', default=50, type=int)
    parser.add_argument('--lr_decay_gamma', help='learning rate decay ratio', default=0.1, type=float)
    parser.add_argument('--loss', help='Segmentation Losses Choices: [ce or focal]', default='ce', type=str)
    parser.add_argument('--use_auxiliary', default=False, type=bool, help='only use in PSPNet')
    

    # set training session
    parser.add_argument('--s', help='training session', default=1, type=int)

    # resume trained model
    parser.add_argument('--r', help='resume checkpoint or not', default=False, type=bool)
    parser.add_argument('--checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', help='checkepoch to load model', default=1, type=int)
    parser.add_argument('--checkpoint', help='checkpoint to load model', default=0, type=int)

    # log and display
    parser.add_argument('--use_tfboard', help='whether use tensorflow tensorboard', default=True, type=bool)

    # configure validation
    parser.add_argument('--no_val', help='not do validation', default=True, type=bool)
    parser.add_argument('--eval_interval', help='iterval to do evaluate', default=1, type=int)

    parser.add_argument('--checkname', help='checkname', default=None, type=str)

    parser.add_argument('--batch_size', help='batch_size', default=2, type=int)
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--crop_size', type=int, default=64, help='crop image size')

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    train_set, val_set, test_set = None, None, None
    # Define Dataloader
    if opt.dataset == 'Cityscapes':
        train_set = Cityscapes.CityscapesSegmentation(opt, split='train')
        val_set = Cityscapes.CityscapesSegmentation(opt, split='val')
        test_set = Cityscapes.CityscapesSegmentation(opt, split='test')
        # opt.base_size = 1024
        # opt.crop_size = 512

        opt.base_size = 128
        opt.crop_size = 64
    if opt.dataset == 'Coco':
        train_set = Coco.COCOSegmentation(opt, split='train')
        val_set = Coco.COCOSegmentation(opt, split='val')
        opt.base_size = 256
        opt.crop_size = 64
    if opt.dataset == 'Pascal':
        train_set = Pascal.VOCSegmentation(opt, split='train')
        val_set = Pascal.VOCSegmentation(opt, split='val')
        test_set = Pascal.VOCSegmentation(opt, split='test')
        opt.base_size = 128
        opt.crop_size = 64

    num_class = train_set.NUM_CLASSES
    kwargs = {'num_workers': opt.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, drop_last=True, **kwargs)

    # Define network
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
        

    # multiple mGPUs
    print("device : ", device)
    if device.type == 'cpu':
        model = torch.nn.DataParallel(model)
    else:
        num_gpus = [i for i in range(opt.n_gpu)]
        model = torch.nn.DataParallel(model, device_ids=num_gpus).cuda()    

    # Define Criterion  
    weight = None
    criterion = SegmentationLosses(weight=weight, cuda=opt.cuda).build_loss(mode=opt.loss)  

    # Define Optimizer
    optimizer = None
    if opt.optimizer == 'adam':
        opt.lr = opt.lr * 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, momentum=0, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0, weight_decay=opt.weight_decay)

    # Define Saver
    saver = Saver(opt)
    saver.save_experiment_config(model.module)


    '''Train'''
    # Resuming checkpoint
    best_pred = 0.0
    lr_stage = [68, 93]
    lr_staget_ind = 0 
    total_loss = 0.0
    model.train()

    total_step = len(train_loader)
    total_train_step = opt.epochs * total_step

    # initialize the early_stopping object
    early_stopping = EarlyStopping(saver, patience=3, verbose=True)

    print('Starting Epoch:', opt.start_epoch)
    print('Total Epoches:', opt.epochs)
    for epoch in range(opt.epochs):
        epoch_loss = []
        with tqdm(total=total_step) as pbar:
            for iteration, batch in enumerate(train_loader):
                imgs, targets = batch['image'], batch['label']
                current_train_step = (epoch) * total_step + (iteration + 1)
                # zero the parameter gradients
                optimizer.zero_grad()

                imgs = Variable(imgs.to(device=device))
                targets = Variable(targets.to(device=device))
                # pbar.set_description("Epoch[{}]({}/{}):img shape={}, learning rate={} ".format(epoch, iteration, len(train_loader), (imgs.shape), opt.lr))
                if opt.use_auxiliary:
                    # PSPNet
                    r = 0.4
                    outputs, aux_outputs = model(imgs) # masks_pred
                    loss = criterion(outputs, targets.long())
                    loss_aux = criterion(aux_outputs, targets.long())
                    loss = loss * (1-r) + loss_aux * r
                else:
                    outputs = model(imgs) # masks_pred
                    loss = criterion(outputs, targets.long())

                pbar.set_description("Model: {}, Loss:{:.4f}, learning rate={} ".format(opt.model, loss.data, optimizer.param_groups[0]["lr"]))
                epoch_loss.append(loss.item())
                # Report Loss
                if (((current_train_step) % 100) == 0) or (current_train_step % 10 == 0 and current_train_step < 100):
                    print("\nepoch: [{}/{}], total step: [{}/{}], batch step [{}/{}], Input shape={}".format(epoch + 1, opt.epochs, current_train_step, total_train_step, iteration + 1, total_step, (imgs.shape)))
                        
                loss.backward(torch.ones_like(loss))
                optimizer.step()
                total_loss += loss.item()
                pbar.update(1)
                break    

        '''Eval'''
        Acc, Acc_class, mIoU, FWIoU = eval_model(model, val_loader)
        print("Acc: {:.5f}, Acc_class: {:.5f}, mIoU: {:.5f}, fwIoU: {:.5f}".format(Acc, Acc_class, mIoU, FWIoU))
        early_stopping(model, optimizer, epoch, Acc) # update patience
        if early_stopping.early_stop:
            print("Early stopping epoch %s"%(epoch))
            break

    