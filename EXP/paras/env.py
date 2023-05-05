"""

"""
import attr
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.EXP.paras.trainer import PLTrainerParas
from HistoMIL.EXP.paras.cohort import CohortParas
from HistoMIL.EXP.paras.dataset import DatasetParas

from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.slides import CollectorParas
@attr.s
class EnvParas(object):
    # for logging and saving
    exp_name:str = None
    # for wandb or other logger
    project:str  = None
    entity:str = None

    # for setup machine
    data_locs:Locations=None
    exp_locs:Locations=None

    # for user
    user_id:str = None

    # for cohort
    main_data_source:str = "slide" # "slide" or "omic"
    collector_para:CollectorParas = CollectorParas()
    cohort_para:CohortParas = CohortParas()
    dataset_para:DatasetParas=DatasetParas()
    # for trainner
    trainer_para:PLTrainerParas = PLTrainerParas()
    opt_para:OptLossParas = OptLossParas()

    



