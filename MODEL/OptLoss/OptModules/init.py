import logging
import torch
import torch.optim as optim
import math
from functools import partial
try:
    from apex.OptLoss.OptModuless import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

try :
    from lars import LARS
    has_lars = True
except ImportError:
    has_lars = False

from HistoMIL.MODEL.OptLoss.OptModules.adafactor import Adafactor
from HistoMIL.MODEL.OptLoss.OptModules.adahessian import Adahessian
from HistoMIL.MODEL.OptLoss.OptModules.adamp import AdamP
from HistoMIL.MODEL.OptLoss.OptModules.nadam import Nadam
from HistoMIL.MODEL.OptLoss.OptModules.radam import RAdam
from HistoMIL.MODEL.OptLoss.OptModules.sgdp import SGDP

pytorch_opt_dict = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'nadam': Nadam,
    'radam': RAdam,
    'adamp': AdamP,
    'sgdp': SGDP,
    'adadelta': optim.Adadelta,
    'adafactor': Adafactor,
    'adahessian': Adahessian,
    'rmsprop': optim.RMSprop,

    #---> gnn
    #'adabound': optim.AdaBound,
    'adagrad': optim.Adagrad,
}

if not has_apex:apex_opt_dict ={}
else:
    apex_opt_dict = {   
        'fused_novograd': FusedNovoGrad,
        'fused_adam': FusedAdam,
        'fused_lamb': FusedLAMB,
        'fused_sgd': FusedSGD,
    }
if not has_lars : lars_opt_dict = {}
else:
    lars_opt_dict = {
        'lars':LARS,
    }

OPTIMIZER_DICTS = {"pytorch":pytorch_opt_dict, 
                    "apex":apex_opt_dict,
                    "lars":lars_opt_dict}
SCHEDULER_DICTS = {
            "steplr":optim.lr_scheduler.StepLR,
            'explr':optim.lr_scheduler.ExponentialLR,
            'multisteplr':optim.lr_scheduler.MultiStepLR,
            'cosinelr':optim.lr_scheduler.CosineAnnealingLR,
            'linearlr':optim.lr_scheduler.LambdaLR,
            'plateaulr':optim.lr_scheduler.ReduceLROnPlateau,
            'maxlr':optim.lr_scheduler.OneCycleLR,
}   