"""
example of run mil experiment
"""
#--------------------------> base env setting
# avoid pandas warning
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# avoid multiprocessing problem
import torch
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')
#--------------------------> logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d|%H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)



#--------------------------> task setting
task_name = "example_mil"
#--------------------------> model setting

from HistoMIL.MODEL.Image.MIL.TransMIL.paras import TransMILParas
# for transmil
model_para_transmil = TransMILParas()
model_para_transmil.feature_size=512
model_para_transmil.n_classes=2
model_para_transmil.norm_layer=nn.LayerNorm
# for dsmil
from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
model_para_dsmil = DSMILParas()
model_para_dsmil.feature_dim = 512 #resnet18
model_para_dsmil.p_class = 2
model_para_dsmil.b_class = 2
model_para_dsmil.dropout_r = 0.5

model_name = "TransMIL"  # or "TransMIL" or "ABMIL"

model_para_settings = {"TransMIL":model_para_transmil,
                       "DSMIL":model_para_dsmil} 



if __name__ == '__main__':

    

    #--------------------------> parameters
    from HistoMIL.EXP.paras.env import EnvParas
    gene2k_env = EnvParas()
    gene2k_env.exp_name = f"{model_name}_{task_name}"
    gene2k_env.project = "gene2k_fast" 
    gene2k_env.entity = "shipan_work"
    #----------------> cohort
    gene2k_env.cohort_para.localcohort_name = "BRCA"
    gene2k_env.cohort_para.task_name = task_name
    gene2k_env.cohort_para.cohort_file = f"/task_name.csv"
    gene2k_env.cohort_para.pid_name = "Patient_ID"
    gene2k_env.cohort_para.targets = [task_name]
    gene2k_env.cohort_para.targets_idx = 0
    gene2k_env.cohort_para.label_dict = {"low":0,"high":1}
    #debug_env.cohort_para.update_localcohort = True
    #----------------> pre-processing
    #----------------> dataset
    gene2k_env.dataset_para.dataset_name = f"BRCA_{task_name}"
    gene2k_env.dataset_para.concepts = ["slide","patch","feature"]
    gene2k_env.dataset_para.split_ratio = [0.8,0.2]
    #----------------> model
    gene2k_env.trainer_para.model_name = model_name
    gene2k_env.trainer_para.model_para = model_para_settings[model_name]
    #----------------> trainer or analyzer
    gene2k_env.trainer_para.backbone_name = "resnet18"
    gene2k_env.trainer_para.additional_pl_paras.update({"accumulate_grad_batches":8})
    gene2k_env.trainer_para.label_format = "int"#"one_hot" 
    #k_fold = None
    #--------------------------> init machine and person
    import pickle
    machine_cohort_loc = "Path/to/BRCA_machine_config.pkl"
    with open(machine_cohort_loc, "rb") as f:   # Unpickling
        [data_locs,exp_locs,machine,user] = pickle.load(f)
    gene2k_env.data_locs = data_locs
    gene2k_env.exp_locs = exp_locs


    #--------------------------> setup experiment
    if __name__ == "__main__":

            logging.info("setup experiment")
            from HistoMIL.EXP.workspace.experiment import Experiment
            exp = Experiment(env_paras=gene2k_env)
            exp.setup_machine(machine=machine,user=user)
            logging.info("setup data")
            exp.init_cohort()
            logging.info("setup trainer..")
            exp.setup_experiment(main_data_source="slide",
                                need_train=True)

            exp.exp_worker.train()


