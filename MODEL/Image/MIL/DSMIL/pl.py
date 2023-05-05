"""
pytorch-lightning wrapper for the model
"""

#---->
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
#---->
from HistoMIL import logger
from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
#---->
####################################################################################
#      pl protocol class
####################################################################################
class  pl_DSMIL(pl_MIL):
    #---->init
    def __init__(self, 
                data_paras:DatasetParas,# dataset para
                opt_paras:OptLossParas,# optimizer para
                trainer_paras:PLTrainerParas,# trainer para
                model_para:DSMILParas):
        super(pl_DSMIL, self).__init__(data_paras,
                                    opt_paras,
                                    trainer_paras,
                                    model_para)
        """
        model:: model instance of abmil
        loss:: name of different loss function
        optimizer:: 
        """
        logger.debug(f"DSMIL pl protocol init done.")
        pass

    def special_loss_calc(self,patch_pred,bag_pred,target):
        # consider which instance is max in the bag
        max_prediction, index = torch.max(patch_pred, 0)
        max_prediction = max_prediction.unsqueeze(0)
        #print(max_prediction.shape, patch_pred.shape,bag_pred.shape, target.shape)
        loss_bag = self.loss(bag_pred, target)
        loss_max = self.loss(max_prediction, target)
        loss_total = 0.5*loss_bag + 0.5*loss_max
        loss = loss_total.mean()
        return loss 
    

    def training_step(self, batch, batch_idx):
        #---->model step
        data, label = batch
        #print(data.shape)
        results_dict = self.model(data)

        #------> check output is valid
        self.confirm_model_outputs(results_dict, self.trainer_paras.model_out_list)
        #---->confirm label format
        label_Y = label
        #print(results_dict,label_Y)
        #---->loss step
        #loss = self.loss(results_dict['logits'], label_Y)
        loss = self.special_loss_calc(patch_pred=results_dict['patch_pred'],
                                       bag_pred= results_dict['logits'],
                                       target= label_Y)

        #---->overall counts log 
        self.counts_step(Y_hat=results_dict['Y_hat'], label=label_Y, train_phase="train")
        return {'loss': loss} 


