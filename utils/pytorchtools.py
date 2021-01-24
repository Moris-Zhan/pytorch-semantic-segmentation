import numpy as np
import torch as T
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, saver = None, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.config = config
        # self.logger = logger 
        # self.vocab = vocab
        # self.title = title 
        self.epoch = 0
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.step = 0
        self.saver = saver

    def __call__(self, model, optimizer, epoch, acc):

        # score = -val_loss

        if self.best_acc is None:
            self.best_acc = acc
            # self.save_checkpoint(val_loss, model)
        elif acc < self.best_acc + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = acc
            # self.save_checkpoint(val_loss, model)
            self.epoch = epoch
            self.save_model(model, optimizer, self.best_acc)
            self.counter = 0 

    # save checkpoint every epoch
    def save_model(self, model, optimizer, best_acc):
        is_best = True
        self.saver.save_checkpoint({
            'epoch': self.epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_acc,
            }, is_best)  