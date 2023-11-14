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
class pl_slide_trainer(pl_base_trainer):
    def __init__(self,
                trainer_para:PLTrainerParas,
                dataset_para:DatasetParas,
                opt_para:OptLossParas,):
        super().__init__(trainer_para,dataset_para,opt_para)
        
    ################################################################
    #   build slide trainner
    ################################################################                       
    def build_model(self):
        logger.info(f"Trainer:: Build model: {self.trainer_para.model_name}:({self.trainer_para.method_type}) with backbone:{self.trainer_para.backbone_name}")
        from HistoMIL.MODEL.Image.init import create_img_model,create_img_mode_paras
        self.model_name = self.trainer_para.model_name # specify algorithm/model for training
        if self.trainer_para.model_para is None:
            logger.info(f"Trainer:: Use default model paras when self.trainer_para.model_para is None")
            self.trainer_para.model_para = create_img_mode_paras(self.trainer_para)
        # init some paras from dataset
        # make sure the model_para is consistent with trainer_para
        self.trainer_para.model_para.class_nb = self.dataset_para.class_nb
        if self.trainer_para.use_pre_calculated :
            logger.info(f"Trainer:: Use pre-calculated feature: {self.trainer_para.model_para.encoder_name}")
            self.trainer_para.model_para.encoder_name = "pre-calculated"
            self.trainer_para.model_para.encoder_pretrained = True
        else:
            logger.info(f"Trainer:: Use backbone: {self.trainer_para.model_para.encoder_name} ")
            self.trainer_para.model_para.encoder_name = self.trainer_para.backbone_name
        # create image model
        logger.info(f"Trainer:: Create image model with paras: {self.trainer_para.model_para}")
        self.pl_model = create_img_model(
                            train_paras=self.trainer_para,
                            optloss_paras=self.opt_para,
                            dataset_paras=self.dataset_para,
                            model_para = self.trainer_para.model_para
                            )

  
        
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
        data_list = slide_list if method_type == "mil" else patch_list

        # is_train related with two things: dataset_para and train_phase
        is_train = self.dataset_para.is_train if self.dataset_para.is_train is not None \
                                         else True if train_phase == "train" else False
        # get dataset
        dataset = create_slide_dataset(
                                data_list=data_list,
                                data_locs=machine.data_locs,
                                concept_paras=collector_para,
                                dataset_paras=self.dataset_para,
                                is_train=is_train,
                                as_PIL=self.dataset_para.as_PIL,
                                )

        return dataset,self.data_cohort.create_dataloader(dataset,
                                                    self.dataset_para)
    

 