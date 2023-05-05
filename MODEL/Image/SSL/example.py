"""
use SSL to train a model
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#--------------------------> parameters
from HistoMIL.EXP.paras.env import EnvParas
debug_env = EnvParas()
debug_env.exp_name = "debug_SSL"
debug_env.project = "test-project" 
debug_env.entity = "shipan_work"
#----------------> cohort
debug_env.cohort_para.localcohort_name = "BRCA"
debug_env.cohort_para.task_name = "DNAD"
debug_env.cohort_para.cohort_file = "/DNAD/DNAD_L2.csv"
debug_env.cohort_para.pid_name = "PatientID"
debug_env.cohort_para.targets = ["HRD"]
debug_env.cohort_para.targets_idx = 0
debug_env.cohort_para.label_dict = {"HRD":0,"HRP":1}
#debug_env.cohort_para.update_localcohort = True
#----------------> pre-processing
#----------------> dataset
debug_env.dataset_para.dataset_name = "DNAD_L2"
debug_env.dataset_para.concepts = ["slide","patch"]
debug_env.dataset_para.split_ratio = [0.8,0.2]

#----------------> model
debug_env.trainer_para.method_type = "patch_learning"
debug_env.trainer_para.model_name = "moco" # 
from HistoMIL.MODEL.Image.SSL.paras import SSLParas
debug_env.trainer_para.model_para = SSLParas()
debug_env.trainer_para.model_para.ssl_dataset_para.batch_size = 64
debug_env.trainer_para.model_para.ssl_dataset_para.label_dict = {"HRD":0,"HRP":1}
debug_env.trainer_para.model_para.ssl_dataset_para.example_file = "example/example.png"
debug_env.trainer_para.model_para.ssl_dataset_para.is_weight_sampler = False
debug_env.trainer_para.model_para.ssl_dataset_para.add_dataloader = {
                                                    "pin_memory":True,
                                                    "drop_last":False,
                                                    }
#----------------> trainer or analyzer
debug_env.trainer_para.label_format = "int"#"one_hot" 
debug_env.trainer_para.additional_pl_paras={
                #---------> paras for pytorch lightning trainner
                "accumulate_grad_batches":4, # mil need accumulated grad
                "accelerator":"auto",#accelerator='gpu', devices=1,
            }
#--------------------------> init machine and person
from HistoMIL.DATA.Cohort.location import Locations
data_locs = Locations(  root=f"/main/PAN/Dataset/{debug_env.cohort_para.localcohort_name}/",
                                sub_dirs={
                                        "slide":f"TCGA-{debug_env.cohort_para.localcohort_name}/",
                                        "tissue":"Tissue/",
                                        "patch":"Patch/",
                                        "patch_img":"Patch_Image/",# need to add for SSL
                                        "feature":"Feature/",
                                        })
exp_locs = Locations(  root="/main/PAN/Exp04_HistoMIL/",
                                sub_dirs={
                                        "src":"HistoMIL/",

                                        "idx":"/Data/",

                                        "saved_models":"/SavedModels/",
                                        "out_files":"/OutFiles/",

                                        "temp":"/Temp/",
                                        "user":"/User/",
                                        
                                     })
debug_env.data_locs = data_locs
debug_env.exp_locs = exp_locs
from HistoMIL.EXP.workspace.env import Machine
machine = Machine(data_locs,exp_locs)
from HistoMIL.EXP.workspace.env import Person
user = Person(id="0001")
user.name = "shi_pan"
user.wandb_api_key = "1266ad70f8bf7695542bf9a2d0dec8748c52431c"




#--------------------------> setup experiment
if __name__ == "__main__":

        logger.info("setup experiment")
        from HistoMIL.EXP.workspace.experiment import Experiment
        exp = Experiment(env_paras=debug_env)
        exp.setup_machine(machine=machine,user=user)
        logger.info("setup data")
        exp.init_cohort()
        logger.info("setup dataset and dataloader..")
        exp.data_cohort.split_train_phase()

        logger.info("setup trainer..")
        from HistoMIL.EXP.trainer.ssl import pl_ssl_trainer
        worker = pl_ssl_trainer(trainer_para=debug_env.trainer_para,
                                dataset_para=debug_env.trainer_para.model_para.ssl_dataset_para,
                                opt_para=debug_env.trainer_para.model_para.ssl_opt_loss_para)
        worker.get_env_info(machine=machine,user=user,
                            project=debug_env.project,entity=debug_env.entity,exp_name=debug_env.exp_name)
        worker.set_cohort(exp.data_cohort)
        worker.build_trainer()
        worker.build_model()
        worker.get_datapack(machine=machine,collector_para=debug_env.collector_para)
        #logger.info("start training..")

        worker.train()

