"""
use pre-process to train a model
"""
# avoid pandas warning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# avoid multiprocessing problem
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from HistoMIL import logger
import logging
logger.setLevel(logging.INFO)

#------>stop skimage warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import imageio.core.util
import skimage 
def ignore_warnings(*args, **kwargs):
    pass
imageio.core.util._precision_warn = ignore_warnings

#--------------------------> parameters
from HistoMIL.EXP.paras.env import EnvParas
preprocess_env = EnvParas()
preprocess_env.exp_name = "debug_preprocess"
preprocess_env.project = "test-project" 
preprocess_env.entity = "shipan_work"
#----------------> cohort
preprocess_env.cohort_para.localcohort_name = "BRCA"
preprocess_env.cohort_para.task_name = "DNAD"
preprocess_env.cohort_para.cohort_file = "/DNAD/DNAD_L2.csv"
preprocess_env.cohort_para.pid_name = "PatientID"
preprocess_env.cohort_para.targets = ["HRD"]
preprocess_env.cohort_para.targets_idx = 0
preprocess_env.cohort_para.label_dict = {"HRD":0,"HRP":1}
#preprocess_env.cohort_para.update_localcohort = True
#----------------> pre-processing

#----------------> model
preprocess_env.trainer_para.method_type = "patch_learning"
preprocess_env.trainer_para.model_name = "moco" # 
from HistoMIL.MODEL.Image.SSL.paras import SSLParas
preprocess_env.trainer_para.model_para = SSLParas()
#----------------> dataset
preprocess_env.dataset_para.dataset_name = "DNAD_L2"
preprocess_env.dataset_para.concepts = ["slide","patch"]
preprocess_env.dataset_para.split_ratio = [0.99,0.01]
#################----> for ssl
preprocess_env.trainer_para.model_para.ssl_dataset_para.batch_size = 16
preprocess_env.trainer_para.model_para.ssl_dataset_para.label_dict = {"HRD":0,"HRP":1}
preprocess_env.trainer_para.model_para.ssl_dataset_para.example_file = "example/example.png"
preprocess_env.trainer_para.model_para.ssl_dataset_para.is_weight_sampler = True
preprocess_env.trainer_para.model_para.ssl_dataset_para.force_balance_val = True
preprocess_env.trainer_para.model_para.ssl_dataset_para.add_dataloader = {
                                                    "pin_memory":True,
                                                    "drop_last":True,
                                                    }

from HistoMIL.DATA.Database.data_aug import SSL_DataAug
# specifu data aug or use default can be found at paras
preprocess_env.trainer_para.model_para.ssl_dataset_para.img_size = (512,512)
add_data_aug_paras = preprocess_env.trainer_para.model_para.ssl_dataset_para.add_data_aug_paras
trans_factory = SSL_DataAug(**add_data_aug_paras)
preprocess_env.trainer_para.model_para.ssl_dataset_para.transfer_fn = trans_factory.get_trans_fn
#----------------> trainer or analyzer
preprocess_env.trainer_para.label_format = "int"#"one_hot" 
preprocess_env.trainer_para.additional_pl_paras={
                #---------> paras for pytorch lightning trainner
                "accumulate_grad_batches":16, # mil need accumulated grad
                "accelerator":"auto",#accelerator='gpu', devices=1,
            }
#preprocess_env.trainer_para.with_logger = None #without wandb to debug
#--------------------------> init machine and person
#--------------------------> init machine and person
import pickle
machine_cohort_loc = "Path/to/BRCA_machine_config.pkl"
with open(machine_cohort_loc, "rb") as f:   # Unpickling
    [data_locs,exp_locs,machine,user] = pickle.load(f)
preprocess_env.data_locs = data_locs
preprocess_env.exp_locs = exp_locs
#--------------------------> setup experiment
if __name__ == "__main__":

        logger.info("setup experiment")
        from HistoMIL.EXP.workspace.experiment import Experiment
        exp = Experiment(env_paras=preprocess_env)
        exp.setup_machine(machine=machine,user=user)
        logger.info("setup data")
        exp.init_cohort()
        logger.info("pre-processing..")
        exp.cohort_slide_preprocessing(concepts=["slide","tissue","patch","feature"],
                                       is_fast=True, force_calc=False)