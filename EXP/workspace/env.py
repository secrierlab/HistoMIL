"""
Related information for person and account
"""
import os

import shutil
import torch

from HistoMIL import logger
from HistoMIL.DATA.FileIO.pkl_worker import save_pkl,load_pkl
from HistoMIL.DATA.Cohort.location import Locations


class Machine:
    """
    Set up envirounment paras related with hardware/device/machine/cluster
    - device{cpu or gpu}
    - data folder
    - experiment folder
    - user related (mainly for logger)
    """
    def __init__(self,
                data_locs:Locations,
                exp_locs:Locations,
                ) -> None:
        # location
        self.data_locs = data_locs
        self.exp_locs  = exp_locs

        self.build_file_tree()
        logger.info("Machine:: Got Machine Parameters")
    ################################################################
    #   Machine related
    ################################################################

    def build_file_tree(self):
        # build file tree system 

        # for omic data data_locs can be None
        if self.data_locs is not None:
            self.data_locs.is_build=True
            self.data_locs.check_structure()

        self.exp_locs.is_build = True
        self.exp_locs.check_structure()

    def get_model_save_dir(self,):
        self.model_save_dir = self.exp_locs.abs_loc("saved_models")
        return self.model_save_dir
    
    def get_temp_dir(self,data_temp:str=None,exp_temp:str=None):
        if data_temp is not None:
            self.data_temp = data_temp
        else:
            self.data_temp = self.data_locs.abs_loc("temp")
        _rm_n_mkdir(loc = self.data_temp)
        if exp_temp is not None:
            self.exp_temp = exp_temp
        else:
            self.exp_temp = self.exp_locs.abs_loc("temp")
        _rm_n_mkdir(loc = self.exp_temp)    

def _rm_n_mkdir(loc:str):
    """Remove and make directory."""
    if not os.path.exists(loc):
        os.makedirs(loc)
    else:
        if os.path.isdir(loc):
            shutil.rmtree(loc)
        os.makedirs(loc)


"""

"""

class Person:
    def __init__(self,id:str):
        #-------> for user save some info
        self.id = id
        self.name = None
        #-------> logger related
        self.wandb_api_key = None

    def loc(self,machine:Machine):
        return machine.exp_locs.abs_loc("user")+self.id+".pkl"

    def read(self,machine:Machine):
        self.person_dict=load_pkl(self.loc())
        self.__setattr__("id",self.person_dict["id"])
        self.__setattr__("name",self.person_dict["name"])
        self.__setattr__("wandb_api_key",self.person_dict["wandb_api_key"])
        

    def save(self,machine:Machine):
        self.person_dict["id"]=self.__getattribute__("id")
        self.person_dict["name"]=self.__getattribute__("name")
        self.person_dict["wandb_api_key"]=self.__getattribute__("wandb_api_key")

        save_pkl(filename=self.loc(machine),save_object=self.person_dict)


