"""
Loss function dictionary

TODO:
    1. fully test all loss functions
"""
#------------> for loss part
import torch
import torch.nn as nn
try:
    from pytorch_toolbelt import losses as L
except ImportError:
    pytorch_toolbelt_flag = False
else:
    pytorch_toolbelt_flag = True
"""
#---------> for different loss
from histocore.MODEL.OptLoss.LossModules.boundary_loss import BDLoss, SoftDiceLoss, DC_and_BD_loss, HDDTBinaryLoss,\
     DC_and_HDBinary_loss, DistBinaryDiceLoss
from histocore.MODEL.OptLoss.LossModules.dice_loss import GDiceLoss, GDiceLossV2, SSLoss, SoftDiceLoss,\
     IoULoss, TverskyLoss, FocalTversky_loss, AsymLoss, DC_and_CE_loss,\
         PenaltyGDiceLoss, DC_and_topk_loss, ExpLog_loss
from histocore.MODEL.OptLoss.LossModules.focal_loss import FocalLoss
from histocore.MODEL.OptLoss.LossModules.hausdorff import HausdorffDTLoss, HausdorffERLoss
from histocore.MODEL.OptLoss.LossModules.lovasz_loss import LovaszSoftmax
from histocore.MODEL.OptLoss.LossModules.ND_Crossentropy import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss,\
     WeightedCrossEntropyLossV2, DisPenalizedCE
"""
from HistoMIL.MODEL.OptLoss.LossModules.coxloss import CoxLoss,CELoss_with_reg,Mixed_CoxLoss_with_reg


pytorch_loss_dict ={
    "CrossEntropyLoss":nn.CrossEntropyLoss,
    "BCEWithLogitsLoss":nn.BCEWithLogitsLoss,
    "BCELoss":nn.BCELoss,
}
pytorch_toolbelt_loss_dict={
   #"focal": L.BinaryFocalLoss,
   #"jaccard":L.BinaryJaccardLoss,
   #"jaccard_log":L.BinaryJaccardLoss,
   #"dice":L.BinaryDiceLoss,
   #"dice_log":L.BinaryDiceLogLoss,
   #"dice_log":L.BinaryDiceLogLoss,
   #"lovasz":L.BinaryLovaszLoss,
   #"bce+lovasz":L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryLovaszLoss(), w1, w2),
   #"bce+jaccard":L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryJaccardLoss(), w1, w2)
   #"bce+log_jaccard":L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryJaccardLogLoss(), w1, w2)
   #"bce+log_dice":L.JointLoss(nn.BCEWithLogitsLoss(), L.BinaryDiceLogLoss(), w1, w2)
}
GAT_loss_dict={
    #https://github.com/TencentAILabHealthcare/MLA-GNN/blob/main/model_GAT_v4.py
        "cox_loss":CoxLoss, #(surv_batch_labels, censor_batch_labels, te_preds)
        "CrossEntropyLoss_with_reg":CELoss_with_reg,
        "MixedCox_CE_with_reg":Mixed_CoxLoss_with_reg,
}

LOSS_DICTS = {  "pytorch":pytorch_loss_dict, 
                "pytorch_toolbelt":pytorch_toolbelt_loss_dict, 
                "GNN_customized":GAT_loss_dict,
                "Custom":{}, # for custom loss
}