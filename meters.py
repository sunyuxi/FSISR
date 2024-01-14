from __future__ import absolute_import

import torch
import datetime
import sys

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def loss_store_init(loss_store):
    """
    initialize loss store, transform list to dict by (loss name -> loss register)
    :param loss_store: the list with name of loss
    :return: the dict of loss store
    """
    dict_store = {}
    for loss_name in loss_store:
        dict_store[loss_name] = AverageMeter()
    loss_store = dict_store
    
    return loss_store

def print_loss(epoch, cur_batch, max_batch, task_name, loss_store=None):
    now = datetime.datetime.now()
    str_now = now.strftime("%Y-%m-%d %H:%M:%S")

    loss_str = task_name + ' ' + str_now + "  epoch: [%3d][%3d/%3d], " % (epoch, cur_batch, max_batch)

    for name, value in loss_store.items():
        loss_str += name + " {:4.3f}".format(value.avg) + "\t"
    print(loss_str)
    sys.stdout.flush()

def reset_loss(loss_store=None):
    for store in loss_store.values():
        store.reset()

def remark_loss(loss_store, *args):
    """
    store loss into loss store by order
    :param args: loss to store
    :return:
    """
    for i, loss_name in enumerate(loss_store.keys()):
        if isinstance(args[i], torch.Tensor):
            loss_store[loss_name].update(args[i].item())
        else:
            loss_store[loss_name].update(args[i])