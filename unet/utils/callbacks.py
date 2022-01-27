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


class LossHistory():
    def __init__(self, model):
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rndm_input = torch.autograd.Variable(torch.rand(1, 3, 512, 512), requires_grad = False).to(device)
        self.writer.add_graph(model, rndm_input)

        f = io.StringIO()
        with redirect_stdout(f):
            summary(model, (3, 512, 512))
        lines = f.getvalue()
        with open(os.path.join(self.log_dir, "summary.txt") ,"w") as f:
            [f.write(line) for line in lines]
        
        os.makedirs(self.save_path)

    def set_status(self, freeze):
        self.freeze = freeze

    def append_loss(self, loss, val_loss, epoch):
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

    def append_loss_no_val(self, loss, epoch):
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
