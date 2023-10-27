from HistoMIL import logger
import random
import pandas as pd
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Cohort.basic import LocalCohort,TaskCohort
from HistoMIL.DATA.Database.dataset import create_slide_dataset
from HistoMIL.EXP.paras.cohort import CohortParas

from HistoMIL.DATA.Database.utils import get_weight_sampler
from HistoMIL.EXP.paras.dataset import DatasetParas


class DataCohort:
    def __init__(self,
                    data_locs:Locations,
                    exp_locs:Locations,
                    cohort_para:CohortParas) -> None:

        self.data_locs = data_locs
        self.exp_locs  = exp_locs
        self.cohort_para = cohort_para

        self.data = None

    def setup_localcohort(self):
        """
        create localcohort obj from folder/files
        in:
            concepts:list: a list of concept that need to be considered
        add:
            self.local_cohort:LocalCohort: obj for a cohort
        """
        slide_root=self.data_locs.abs_loc("slide")
        concepts = self.cohort_para.local_cohort_concepts
        concepts_para = self.cohort_para.local_cohort_concepts_para
        is_update = self.cohort_para.update_localcohort
        file_pattern = self.cohort_para.slide_pattern
        logger.info(f"Cohort::Set up local cohort for slides at {slide_root}")
        self.local_cohort=LocalCohort(slide_root=slide_root,
                                      idx_root=self.exp_locs.abs_loc("idx"),
                                      cohort_name=self.cohort_para.localcohort_name,
                                      concepts=concepts,
                                      concept_paras=concepts_para)
        loc = self.local_cohort.loc()
        if Path(str(loc)).exists() and not is_update:
            self.local_cohort.read()
        else:
            logger.info(f"Cohort::Build local cohort use {slide_root}")
            self.local_cohort.build(data_locs=self.data_locs,pattern=file_pattern)

    def setup_taskcohort(self,
                        #---->related with concept
                        task_concepts:list = ["slide","patch"],#from dataset_para
                        ):
        """
        create taskcohort obj from csv
        in:
            concepts:list: a list of concept that need to be considered in exp
        add:
            self.cohort:Cohort: obj for a cohort
        """

        assert self.cohort_para.cohort_file is not None
        assert self.cohort_para.pid_name is not None
        assert self.cohort_para.targets is not None
        csv_loc = self.cohort_para.cohort_file
        task_name = self.cohort_para.task_name
        is_update = self.cohort_para.update_taskcohort

        logger.info(f"Cohort::Set up task cohort for file {csv_loc}")
        self.task_cohort = TaskCohort(
                                    task_name=task_name,
                                    idx_folder=Path(self.exp_locs.abs_loc("idx")),
                                    cohort_paras=self.cohort_para,
                                    task_concepts=task_concepts,
                                    concept_paras = self.cohort_para.taskcohort_concepts_para,
                                    )

        
        loc = self.task_cohort.loc()
        if Path(str(loc)).exists() and not is_update:
            logger.info(f"Cohort::Read task cohort from {loc}")
            self.task_cohort.read()
        else:
            # build task cohort with the slides that have concepts file in concepts list
            logger.info(f"Cohort::Build task cohort use {csv_loc}")
            self.task_cohort.build(local_df=self.local_cohort.usable_df(task_concepts))
            logger.info(f"Cohort::Done and task cohort saved as {loc}")

    def cohort_shuffle(self):
        if self.cohort_para.is_shuffle:
            logger.info(f"Cohort::Shuffle in dataframe level or Cohort level.")
            df = self.task_cohort.table.df
            df = df.sample(frac=1).reset_index(drop=True)
            self.task_cohort.table.df = df
        else:
            logger.info("Cohort::Not shuffle in dataframe level or Cohort level.")
            
    def show_taskcohort_stat(self,label_idx:str="HRD",concept_name:str="patch"):
        cat_stat = self.task_cohort.get_concepts_stat(label_idx=label_idx,
                                                      concept_name=concept_name)
        label_dict = self.task_cohort.cohort_paras.label_dict
        logger.info(f" Cohort::Show Task stat with {label_dict}:")
        for i in range(len(cat_stat)):
            logger.info(f" Cohort::Category: {cat_stat[i][0]} include {len(cat_stat[i][1])} slides,")
            logger.info(f"               include {sum(cat_stat[i][1])}  {concept_name}, ")
        return cat_stat

    def split_train_phase(self,
                        ratio:list=[0.8,0.2],#in dataset_para
                        label_name:str="HRD",    #select one category name
                        K_fold:int=None,target_df=None):
        # target_df can be customized 
        target_df = self.task_cohort.table.df if target_df is None else target_df
        test_size = 1-ratio[0]
        if K_fold is None:
            logger.warning("Cohort::Using ratio split.")
            from sklearn.model_selection import train_test_split   
            train_data,test_data = train_test_split(target_df, 
                                        test_size=test_size,
                                        stratify=target_df[label_name])
            self.data = {"all_df":target_df,"train":train_data}
            self.data.update({"test":test_data})
        else:
            logger.warning("Cohort::Using k-fold split rather than ratio.")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=K_fold,shuffle=True,random_state=2022)
            idx_lists = list(kf.split(target_df))
            self.get_k_th_fold(target_df,idx_lists,label_name,i_th=0)
        
            

    def get_k_th_fold(self,df:pd.DataFrame,idx_lists:list,target_label:str,i_th:int=0):
            
        df_train = df.iloc[idx_lists[i_th][0].tolist()]
        df_test = df.iloc[idx_lists[i_th][1].tolist()]

        self.data = {"all_df":df,"train":df_train,"test":df_test,"idxs":idx_lists}

    def get_task_datalist(self,phase:str="train"):
        """
        from a cohort select slides that have concept in concepts list, then build lists
        in:
            target_df:pandas.DataFrame: target dataframe that 
        out:
            slide_list: list for MIL training
            patch_list: list for patch-level training
            patch_nb:   list for patch numbers 
        """
        target_df = self.data[phase]
        slide_list,patch_list,patch_nb = self.task_cohort.get_task_datalist(dataframe=target_df)
        return slide_list,patch_list,patch_nb
    
    def build_dataset(self,data_list:list,dataset_paras:DatasetParas,is_train:bool=True,is_shuffle:bool=True):
        """
        build dataset for different training models
        in:
            folders:list,fnames:list,labels:list,patch_nbs:list : output of cohort2filelist
        out:
            datalist:list: direct input to MILSet
            min_patch_nb:int: the min number of patches which is the maximun nb of sample_nb 
        """
        return create_slide_dataset(data_locs=self.data_locs,
                                data_list=data_list,
                                concept_paras=self.task_cohort.concept_paras,
                                dataset_paras=dataset_paras,
                                is_train=is_train,
                               )

    def create_dataloader(self,dataset,dataset_para:DatasetParas):
        """
        complex with bug fixing. need to be improved
        """
        #construct paras for dataloader 
        dataloader_para = {"batch_size":dataset_para.batch_size}
        if dataset_para.is_weight_sampler:
            #logger.info(f"Using weight sampler with {dataset.label_dict}")
            if len(dataset.data[0]) == 3:
                L = [dataset.label_dict[l] for [_,_,l] in dataset.data]
            elif len(dataset.data[0]) == 4:
                L = [dataset.label_dict[l] for [_,_,_,l] in dataset.data]
            else:
                raise ValueError("The dataset is not correct.")
            label_np = np.array(L)
            logger.info(f"Cohort:: For weight sampler, the label distribution is {np.unique(label_np,return_counts=True)}")
            sampler = get_weight_sampler(dataset,label_np)
            dataloader_para.update({"sampler":sampler})
        else:
            dataloader_para.update({"shuffle":dataset_para.is_shuffle})
        if dataset_para.num_workers >1:
            dataloader_para.update({"num_workers":dataset_para.num_workers})
        if dataset_para.add_dataloader is not None:
            dataloader_para.update(dataset_para.add_dataloader)
            """
            pin_memory=self.hparams.pin_data_memory,
            drop_last=self.hparams.drop_last_batch,
            """
        logger.info(f"Cohort::Dataloader paras as {dataloader_para}")
        #construct dataloader
        dataloader = DataLoader(dataset, 
                                **dataloader_para)
                

        return dataloader
