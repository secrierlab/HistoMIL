import os
import numpy as np
import pytorch_lightning as pl

import torch
from pathlib import Path

from HistoMIL import logger
from HistoMIL.EXP.workspace.env import Machine
from HistoMIL.EXP.paras.slides import CollectorParas
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.DATA.Cohort.data import DataCohort
from HistoMIL.DATA.Database.dataset import create_slide_dataset


class pl_base_trainer:
    """
    base class for pl trainer pipeline
    """
    def __init__(self,
                trainer_para:PLTrainerParas,
                dataset_para:DatasetParas,
                opt_para:OptLossParas,):
        
        self.trainer_para = trainer_para 
        self.dataset_para = dataset_para
        self.opt_para = opt_para

        self.data_pack = {}
        self.pl_model = None

        self.machine = None
        self.user = None
        self.project = None
        self.entity = None
        self.exp_name = None

    ################################################################
    #   build common function trainner
    ################################################################ 
    def get_env_info(self,machine,user,project,entity,exp_name):
        self.machine = machine
        self.user = user
        self.project = project
        self.entity = entity
        self.exp_name = exp_name
        
    def build_trainer(self):
        trainer_additional_dict = self.trainer_para.additional_pl_paras
        callbacks_list = []

        # 4. create learning rate logger
        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks_list.append(lr_monitor)
        if self.trainer_para.with_logger=="wandb":
            # 4. Create wandb logger
            from pytorch_lightning.loggers import WandbLogger
            os.environ["WANDB_API_KEY"]=self.user.wandb_api_key

            wandb_logger = WandbLogger(project=self.project,
                                        entity=self.entity,
                                        name=self.exp_name)
            trainer_additional_dict.update({"logger":wandb_logger})

        if self.trainer_para.with_ckpt:
            # 4. check point
            from pytorch_lightning.callbacks import ModelCheckpoint
            ckpt_paras = self.trainer_para.ckpt_para
            ckpt_name = self.exp_name+self.trainer_para.ckpt_format
            logger.debug(f"Trainer:: for exp {self.exp_name} Checkpoint with paras {ckpt_paras}")
            checkpoint_callback = ModelCheckpoint(dirpath=self.machine.exp_locs.abs_loc("saved_models"),
                                                filename=ckpt_name,
                                                 **ckpt_paras,
                                                )
            ckpt_dir = self.machine.exp_locs.abs_loc("saved_models")
            logger.info(f"Trainer:: Best model will be saved at {ckpt_dir} as {ckpt_name}")
            callbacks_list.append(checkpoint_callback)
            
        if len(callbacks_list)>=1: trainer_additional_dict.update(
                                            {"callbacks":callbacks_list})
        # 4. Trainer and fit
        self.trainer = pl.Trainer(default_root_dir=self.machine.exp_locs.\
                                                        abs_loc("out_files"),
                                max_epochs=self.opt_para.max_epochs,
                                **trainer_additional_dict
                                )

    def train(self):
        logger.info("Trainer:: Start training....")
        trainloader = self.data_pack["trainloader"]
        valloader = self.data_pack["testloader"]
        self.trainer.fit(model=self.pl_model, 
                train_dataloaders=trainloader,
                val_dataloaders=valloader)

    def validate(self,ckpt_path:str="best"):
        testloader = self.data_pack["testloader"]
        out = self.trainer.validate(dataloaders=testloader, ckpt_path=ckpt_path,)
        return out
    ################################################################
    #   cohort related function
    ################################################################ 
    def set_cohort(self,data_cohort:DataCohort):
        self.data_cohort = data_cohort

    def dataloader_init_fn(self,train_phase:str,
                            machine:Machine,
                            colloctor_para:CollectorParas,):
        raise NotImplementedError

    def change_label_dict(self,dataset,dataloader):
        # get original label dict
        label_dict = self.data_cohort.cohort_para.label_dict
        # get one example label and transfer to np array
        l_example = list(label_dict.keys())[0]
        if type(l_example) != list:l_example = [l_example] # make sure it is a list otherwise it will not have shape
        label_example = np.array(l_example) # convert to np array

        # check target format
        target_format = self.trainer_para.label_format

        # transfer label dict
        from HistoMIL.MODEL.Image.PL_protocol.utils import current_label_format,label_format_transfer
        current_format = current_label_format(label_example,task=self.trainer_para.task_type)
        if current_format!=target_format:
            target_dict = label_format_transfer(target_format,label_dict)
            dataset.label_dict = target_dict
            dataloader.dataset.label_dict = target_dict
            logger.info(f"Trainer:: Change label into: {target_dict}")


        return dataset,dataloader

    def get_datapack(self,
                    machine:Machine,
                    collector_para:CollectorParas,):
        """
        create self.datapack which include trainset,testset,
        trainloader, testloader
        in:
            machine:Machine: machine object for data path
            collector_para:CollectorParas: paras for data collector

        """
        is_shuffle = self.dataset_para.is_shuffle
        is_weight_sampler = self.dataset_para.is_weight_sampler
        #---> for train phase
        trainset,trainloader = self.dataloader_init_fn(train_phase="train",
                                            machine=machine,
                                            collector_para=collector_para)

        #---> for validation phase
        if not self.dataset_para.force_balance_val:
            self.dataset_para.is_shuffle=False # not shuffle for validation
            self.dataset_para.is_weight_sampler=False
        testset,testloader = self.dataloader_init_fn(train_phase="test",
                                            machine=machine,
                                            collector_para=collector_para)
        
        #---> setup dataset meta
        self.dataset_para.data_len = trainset.__len__()
        _,dict_L = trainset.get_balanced_weight(device="cpu")
        self.dataset_para.category_ratio = dict_L

        #----> change back for next run
        self.dataset_para.is_shuffle=is_shuffle # not shuffle for validation
        self.dataset_para.is_weight_sampler=is_weight_sampler
        
        #----> change label_dict to fit the model and loss
        # get original label dict
        trainset,trainloader = self.change_label_dict(trainset,trainloader)
        testset,testloader = self.change_label_dict(testset,testloader)
        #----> save to self
        self.data_pack = {
            "trainset":trainset,
            "trainloader":trainloader,
            "testset":testset,
            "testloader":testloader,
        }
