"""
pytorch-lightning wrapper for the model
"""

#---->
import pytorch_lightning as pl

#---->
from HistoMIL import logger
from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.MODEL.Image.MIL.TransMIL.paras import TransMILParas
#---->
####################################################################################
#      pl protocol class
####################################################################################
class  pl_TransMIL(pl_MIL):
    #---->init
    def __init__(self, 
                data_paras:DatasetParas,# dataset para
                opt_paras:OptLossParas,# optimizer para
                trainer_paras:PLTrainerParas,# trainer para
                model_para:TransMILParas,# model para
                ):
        super(pl_TransMIL, self).__init__(data_paras,
                                    opt_paras,
                                    trainer_paras,
                                    model_para)
        """
        model:: model instance of tran_mil
        loss:: name of different loss function
        optimizer:: 
        """
        logger.info("TransMIL pl protocol init done.")
        pass

    def infer_step(self, batch, batch_idx):
        """
        designed for inference and get heatmap of a slide
        """
        data, label = batch
        #print(data.shape)
        results_dict = self.model(data)
        att = results_dict["att"]
        results_dict["att"] = att.detach().cpu().numpy()

        return results_dict