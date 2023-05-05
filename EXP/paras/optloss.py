"""
paras for optimizer and loss function
"""

import attr
import torch

@attr.s(auto_attribs=True)
class OptLossParas(object):
    #----> loss meta
    Loss_name: list = ["CrossEntropyLoss",]
    Loss_lib:list   = ["pytorch",]

    #----> loss weight
    imbalance_weighted_loss:bool=False
    imbalance_weight:torch.Tensor=None # can be given from DatasetParas.imbalance_loss_weight

    multiple_loss:bool=False
    multiple_loss_weights:list = None # a list of weights for each loss

    #----> opt meta
    Opt_pkg: str = "pytorch"
    Opt_name: str = "adam"
    lr: float = 0.001
    opt_paras: dict = {}

    #----> opt scheduler para
    Scheduler_name: str = "steplr"
    max_iter: int = 100
    scheduler_paras: dict = {"step_size": 10, 
                            #"gamma": 0.1,
                            }
    max_epochs: int = 100
    #----> train related
    shuffle: bool = True


    





