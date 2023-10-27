"""
class for cohort parameters

localcohort
taskcohort
"""
from HistoMIL.DATA.Slide.collector.data_collector import default_concept_keys
from HistoMIL.EXP.paras.slides import DEFAULT_CONCEPT_PARAS, CollectorParas
##############################################################################
#           para for cohort
##############################################################################

class CohortParas(object):
    """
    parameters used for building cohort for task and dataset
    localcohort
    taskcohort
    """
    def __init__(self) -> None:

        # local_cohort: get slide info from folder
        self.localcohort_name:str=None
        self.slide_pattern:str="*.svs"  #::str:: file pattern for slide files
        self.local_cohort_concepts:list=default_concept_keys #::list:: concepts for local cohort
        self.local_cohort_concepts_para:CollectorParas=DEFAULT_CONCEPT_PARAS #::CollectorParas:: paras for local cohort
        self.update_localcohort:bool=False #::bool:: if update local cohort

        # task_cohort: get slide info from database
        # task related meta info
        self.task_name:str = None       #::str:: name of the task
        self.update_taskcohort:bool=True #::bool:: if update task cohort
        self.taskcohort_concepts_para:CollectorParas=DEFAULT_CONCEPT_PARAS
        # label csv
        self.cohort_file:str=None  #::str:: csv file for possible label 
        self.cohort_data_file:str=None  #::str:: csv file for possible data matrix link with pid (GeneExpressionProfile)
        
        self.pid_name:str=None     #"PatientID"  #::str:: category name for PatientID
        self.targets:list=None     #  ["target1","target2"]  #::list::  list of labels can read from csv
        self.targets_idx:int=0    #::int:: index of target in targets list when you only need one
        self.task_additional_idx:str = None # additional content into task csv files
        # for cohort with r file
        self.r_key:str = None          #::str:: key for cohort in r file    
        self.with_transpose:bool = False #::bool:: if transpose the cohort matrix

        self.label_dict:dict={"label1":0,"label2":1}  #::dict:: a dict for possible labels and related int
        self.category_nb:int=len(self.label_dict.keys())  #::int:: number of classes
        self.is_shuffle:bool=True  #::bool:: if shuffle the cohort level(or shuffle in df)