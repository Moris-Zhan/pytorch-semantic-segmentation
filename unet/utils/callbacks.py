import os
import torch
import scipy.signal
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torchsummary import summary
import io
from contextlib import redirect_stdout
import threading
import webbrowser
import numpy as np
from copy import deepcopy


class LossHistory():
    def __init__(self, model, patience = 5):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = "logs//UNet/"
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "run_" + str(self.time_str)))        
        self.freeze = False

        # write model summary
        x = threading.Thread(target=self.write_summary, args=([deepcopy(model.module).cpu()]))
        x.start() 

        # launch tensorboard
        t = threading.Thread(target=self.launchTensorBoard, args=([self.log_dir]))
        t.start()     

        # initial EarlyStopping
        self.patience = patience
        self.reset_stop()
           
        os.makedirs(self.save_path)
    
    def write_summary(self, cpu_model):
        print("write model summary ready")
        rndm_input = torch.autograd.Variable(torch.rand(1, 3, 600, 600), requires_grad = False).cpu()
        self.writer.add_graph(cpu_model, rndm_input)

        print("tensroboard model summary finished")

        f = io.StringIO()
        with redirect_stdout(f):
            summary(cpu_model, (3, 600, 600), device="cpu")
        lines = f.getvalue()
        with open(os.path.join(self.log_dir, "summary.txt") ,"w") as f:
            [f.write(line) for line in lines]

        print("write model summary finished")
        return

    def launchTensorBoard(self, tensorBoardPath, port = 8888):
        os.system('tensorboard --logdir=%s --port=%s'%(tensorBoardPath, port))
        url = "http://localhost:%s/"%(port)
        # webbrowser.open_new(url)
        return

    def reset_stop(self):
        self.best_epoch_loss = np.Inf 
        self.stopping = False
        self.counter  = 0

    def set_status(self, freeze):
        self.freeze = freeze

    def epoch_loss(self, loss, val_loss, epoch):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.loss_plot()

        prefix = "Freeze_epoch/" if self.freeze else "UnFreeze_epoch/"     
        self.writer.add_scalar(prefix+'Loss/Train', loss, epoch)
        self.writer.add_scalar(prefix+'Loss/Val', val_loss, epoch)  
        self.decide(val_loss)

    def epoch_loss_no_val(self, loss, epoch):
        self.losses.append(loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")       

        self.loss_plot()

        prefix = "Freeze_epoch/" if self.freeze else "UnFreeze_epoch/"     
        self.writer.add_scalar(prefix+'Loss/Train', loss, epoch)  

    def step(self, steploss, stepfscore, iteration):        
        prefix = "Freeze_step/" if self.freeze else "UnFreeze_step/"
        self.writer.add_scalar(prefix + 'Train/Loss', steploss, iteration)
        self.writer.add_scalar(prefix + 'Train/F_Score', stepfscore, iteration)
    
    def decide(self, epoch_loss):
        if epoch_loss > self.best_epoch_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'Best lower loss:{self.best_epoch_loss}')
                self.stopping = True
        else:
            self.best_epoch_loss = epoch_loss           
            self.counter = 0 
            self.stopping = False

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")
