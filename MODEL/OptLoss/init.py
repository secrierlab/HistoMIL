import logging
#------------> for optimizer part
import torch
import torch.optim as optim
import math
from functools import partial

#---------> for init functions
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas

############################################################################################################
# .      create_loss and create_optim
############################################################################################################
from HistoMIL.MODEL.OptLoss.LossModules.init import LOSS_DICTS,pytorch_toolbelt_flag
from HistoMIL.MODEL.OptLoss.OptModules.init import OPTIMIZER_DICTS,has_lars,has_apex,SCHEDULER_DICTS
############################################################################################################

############################################################################################################
# .      create_loss and create_optim
############################################################################################################
class OptLossFactory:
    def __init__(self,
                para:OptLossParas, 
                data_paras:DatasetParas) -> None:
        """
        create loss and optimizer instance and keep meta info
        """
        self.para = para
        self.data_paras = data_paras

        #--------> possible loss
        self.LOSS_DICTS = LOSS_DICTS
        self.pytorch_toolbelt_flag = pytorch_toolbelt_flag

        #--------> with loss weight
        self.imbalance_weight = self.get_imbalance_weight()
        self.loss=None # for each trianing need to explicitly define loss

        #--------> possible optimizer and scheduler
        self.OPTIMIZER_DICTS = OPTIMIZER_DICTS
        self.SCHEDULER_DICTS = SCHEDULER_DICTS

    
    def get_imbalance_weight(self):
        if self.para.imbalance_weighted_loss==True:
            # given or calculated
            if self.para.imbalance_weight is not None: weight = self.para.imbalance_weight
            else:  
                weight = BCE_find_weights(self.data_paras)
                self.para.imbalance_weight = weight
        else:
            weight=None
        return weight

    def _create_single_loss(self,Loss_name:str,Loss_lib:str):
        """
        create loss function instance for pytorch-lightning module
        :param para: paras for loss
        :param data_paras: paras for dataset
        :return: loss function instance
        """
        #--------> make sure it works
        assert Loss_lib in self.LOSS_DICTS.keys(), f"Invalid loss lib {Loss_lib}"
        if (Loss_lib=="pytorch_toolbelt") and (not self.pytorch_toolbelt_flag):
            raise ImportError("pytorch_toolbelt is not installed")

        #--------> select loss
        loss = None
        target_loss_dict = self.LOSS_DICTS[Loss_lib]
        if Loss_lib == "GNN_customized":
            # GNN customized loss only include function not a class
            logging.debug(f"GNN loss function {Loss_name} from {Loss_lib}")
            loss = target_loss_dict[Loss_name]
        else:
            logging.debug(f"Normal Loss function {Loss_name} from {Loss_lib}")
            if Loss_name in target_loss_dict.keys():
                if self.imbalance_weight is not None: 
                    loss = target_loss_dict[Loss_name](self.imbalance_weight)
                else: loss = target_loss_dict[Loss_name]()
            else:
                raise ValueError(f"Invalid loss name {Loss_name}")
        return loss
    
    def _create_join_loss(self,loss1,loss2,w1,w2):
        """
        create loss function instance for pytorch-lightning module
        loss = w1*loss1 + w2*loss2
        """
        if  self.imbalance_weight is None and pytorch_toolbelt_flag:
            from pytorch_toolbelt import losses as L
            loss = L.JointLoss(loss1, loss2, w1, w2)
        else:
            raise ValueError(f"Can not create joint loss with \
                    self.imbalance_weight={self.imbalance_weight} and \
                    pytorch_toolbelt_flag {pytorch_toolbelt_flag}")
        return loss
    
    def create_loss(self):
        logging.info(f"Create loss function {self.para.Loss_name}")
        if self.para.multiple_loss==False:
            self.loss = self._create_single_loss(self.para.Loss_name[0],\
                                                    self.para.Loss_lib[0])
        else:
            loss1 = self._create_single_loss(self.para.Loss_name[0],
                                            self.para.Loss_lib[0])
            loss2 = self._create_single_loss(self.para.Loss_name[1],
                                            self.para.Loss_lib[1])
            self.loss = self._create_join_loss(loss1,loss2,
                                        self.para.multiple_loss_weights[0],
                                        self.para.multiple_loss_weights[1])
        logging.debug(self.loss)

    def create_optimizer(self,model_para):
        """
        create optimizer instance for pytorch-lightning module
        :param para: paras for optimizer
        :return: optimizer instance
        """
        #--------> make sure it works
        if (self.para.Opt_name=="LARS") and (not has_lars):
            raise ImportError("LARS is not installed")
        if (self.para.Opt_name=="apex") and (not has_apex):
            raise ImportError("apex is not installed")

        #--------> select optimizer
        optimizer = None
        if self.para.Opt_pkg in self.OPTIMIZER_DICTS.keys():
            pkg_name = self.para.Opt_pkg
            optimizer = self.OPTIMIZER_DICTS[pkg_name][self.para.Opt_name](model_para,
                                                            lr=self.para.lr,
                                                            **self.para.opt_paras)
        else:
            raise ValueError(f"Invalid optimizer name {self.para.Opt_name}")
        return optimizer

    def create_scheduler(self,optimizer):
        """
        create scheduler instance for pytorch-lightning module
        :optimizer: optimizer instance
        :return: scheduler instance
        """
        scheduler = None
        if self.para.Scheduler_name in self.SCHEDULER_DICTS.keys():
            scheduler = self.SCHEDULER_DICTS[self.para.Scheduler_name](optimizer,
                                                            **self.para.scheduler_paras)
        else:
            raise ValueError(f"Invalid scheduler name {self.para.Scheduler_name}")
        return scheduler



############################################################################################################
# .      functions for create_loss
############################################################################################################
def BCE_find_weights(data_paras:DatasetParas):
    class_dict = data_paras.label_dict
    list_of_key = list(class_dict.keys())
    list_of_value = list(class_dict.values())
    # only int can be used for idx of list
    assert type(list_of_value[0]) == int
    w_list = []
    for i in range(len(list_of_key)):
        w_list.append(data_paras.category_ratio[list_of_key[list_of_value.index(i)]])
    #w1=data_paras.category_ratio[list_of_key[list_of_value.index(0)]]
    #w2=data_paras.category_ratio[list_of_key[list_of_value.index(1)]]
    return torch.tensor(w_list)




############################################################################################################
# .      functions for create_optimizer
############################################################################################################
