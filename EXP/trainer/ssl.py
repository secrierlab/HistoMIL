"""
create pl slide trainer from paras
"""
import os
import pytorch_lightning as pl

import torch
from pathlib import Path

from HistoMIL import logger
from HistoMIL.EXP.workspace.env import Machine
from HistoMIL.EXP.paras.slides import CollectorParas
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.DATA.Database.dataset import create_slide_dataset
from HistoMIL.EXP.trainer.base import pl_base_trainer
##########################################################################################
#   slide trainer related function
##########################################################################################
class pl_ssl_trainer(pl_base_trainer):
    def __init__(self,
                trainer_para:PLTrainerParas,
                dataset_para:DatasetParas,
                opt_para:OptLossParas,):
        super().__init__(trainer_para,dataset_para,opt_para)
        
    ################################################################
    #   build slide trainner
    ################################################################                       
    def build_model(self):
        logger.info(f"Trainer SSL:: Build model: {self.trainer_para.model_name}:({self.trainer_para.method_type}) with backbone:{self.trainer_para.backbone_name}")
        # create image model
        from HistoMIL.MODEL.Image.SSL.pl import pl_SSL
        self.pl_model = pl_SSL(
                        model_paras=self.trainer_para.model_para)

  
        
    ################################################################
    #   data loader from cohort
    ################################################################
    def dataloader_init_fn(self,
                        train_phase:str,#"train",or "test"
                        machine:Machine,
                        collector_para:CollectorParas,
                        ):
        """
        produce dataset and dataloader for different training phase
        """
        slide_list,patch_list,_ = self.data_cohort.get_task_datalist(phase=train_phase)

        # different slide methods need different training protocol 
        # data_list related with training methods,
        # (1)mil need slide list 
        # (2)and transfer learning need patch list
        method_type = self.trainer_para.method_type# "mil" or "patch_learning"
        assert method_type in ["mil","patch_learning"]


        # is_train related with two things: dataset_para and train_phase
        is_train = self.dataset_para.is_train if self.dataset_para.is_train is not None \
                                         else True if train_phase == "train" else False
        # get dataset
        dataset=self.data_cohort.build_dataset(data_list=patch_list,
                                    dataset_paras=self.trainer_para.model_para.ssl_dataset_para,
                                    is_train=is_train)
        dataloader = self.data_cohort.create_dataloader(dataset=dataset,
                                    dataset_para=self.trainer_para.model_para.ssl_dataset_para)


        return dataset,dataloader
    

 