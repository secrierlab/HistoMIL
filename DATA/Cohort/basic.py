"""
Integrate informtion from CSV files
"""
import random
from HistoMIL.EXP.paras.cohort import CohortParas
import numpy as np
from HistoMIL import logger
import pandas as pd
from pathlib import Path

from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Slide.collector.data_collector import WSICollector,default_concept_keys
from HistoMIL.EXP.paras.slides import DEFAULT_CONCEPT_PARAS,CollectorParas
from HistoMIL.DATA.FileIO.table_backends import tableWorker

class Cohort:
    """
    include basic functions for cohorts
    """
    def __init__(self,idx_root:Path,
                    r_key = None,
                    concept_paras:CollectorParas=DEFAULT_CONCEPT_PARAS) -> None:
        self.idx_root = idx_root
        self.r_key = r_key

        self.concept_paras = concept_paras

    def init_table(self,loc:str):
        self.cohort_loc = loc
        self.table = tableWorker(loc = self.cohort_loc)

    def read(self):
        self.table.read_table(key = self.r_key)
        
    ###############################################################################
    #           for local_cohort 
    ###############################################################################

    def local_slides(self,
                    slide_root:str,
                    pattern:str="*.svs",
                    concepts=default_concept_keys):
        """
        collect all slides in local machine given folder, make a panda df
        in:
            slide_root:str: root location for possible slides downloaded 
                            from TCGA source e.g. /TCGA-BRCA/{folder}/{filename}.svs
            pattern:str: pattern string for rglob function to select files e.g."*.svs"
            concepts:list: a list of concepts
        out:
            self.table.df::pandas.DataFrame
        """
        logger.debug(f"Cohort::Checking local slide files  {pattern} at root loc {slide_root}")
        # list downloaded slide dataset and create a csv for it
        filenames = []
        folders   = []
        P_id      = []

        p = Path(slide_root)
        for f in p.rglob(pattern):
            filenames.append(f.name)
            folders.append(f.parts[-2])
            P_id.append(f.stem[:12])
   
        assert len(filenames)>0
        data_dict = {
                "PatientID":P_id,
                "folder":folders,
                "filename":filenames,
                }
        for name in concepts:
            zero_list = [0 for i in P_id]
            data_dict.update({name:zero_list})
        #return pd.DataFrame(data_dict)
        self.table.df = pd.DataFrame(data_dict)

    def add_concept_to_df(self,concepts):
        for name in concepts:
            if name not in self.table.df.keys():
                zero_list = [0 for i in range(self.table.df.shape[0])]
                self.table.df[name] = zero_list
            nb_name = name+"_nb"
            if nb_name not in self.table.df.keys():
                zero_list = [0 for i in range(self.table.df.shape[0])]
                self.table.df[nb_name] = zero_list

    def check_concepts(self,data_locs:Locations,concepts:list):
        """
        check usable concepts files and save it into cohort_df
        in:
            data_locs::Locations:: data location obj, include data folders
        change:
            self.cohort_df::pandas.DataFrame:: data table for later process
        """
        self.table.df["patch_nb"]=0
        for i in range(self.table.df.shape[0]):
            folder = self.table.df["folder"][i]
            fname = self.table.df["filename"][i]
            C = WSICollector(db_loc=data_locs,wsi_loc=folder+"/"+fname,paras=self.concept_paras)
            for name in concepts:
                # feature need to specify model_name
                C_flag,C_nb = C.with_saved_data(name=name)
                #print(f" {fname} concept {name} is {C_flag}")
                self.table.df[name][i]=C_flag
                self.table.df[name+"_nb"][i]=C_nb
        self.table.update_loc(self.cohort_loc)
        self.table.write_table()

    ###############################################################################
    #           for task_cohort
    ###############################################################################

    def sort_patientID(self,table:tableWorker,pid_name:str,labels_name:list):
        """
        sort input csv into a format with PatientID and others in labels_name list
        in:
            input_csv:Path: an csv with some format
            pid_name:str:   the name of patientID in this csv
            labels_name:list other important part in this csv
        """
        # read an un-formatted csv and format it with PatientID and other labels_name list
        P_id =  table.df[pid_name].str[:12].values.tolist()

        data_dict = {
                "PatientID":P_id,
        }
        for name in labels_name:
            data_dict.update({name:table.df[name].values.tolist()})
        
        new_df = pd.DataFrame(data_dict)
        return new_df
  
    def merge_with_PID(self,df_1,df_2):
        """
        create cohort table for a label
        in:
            df_1::pandas.DataFrame:: cohort table for a label
            df_2::pandas.DataFrame:: cohort table for a label
        out:
            df::pandas.DataFrame:: merged cohort table
        """
        merged_df=pd.merge(df_1,df_2,on=["PatientID"])#),how='left')
        return merged_df

    def get_dataloader_lists(self,df,concepts:list,label_name:str):
        """
        get a filelist and confirm all element include the needed concepts
        in:
            df::pandas.DataFrame:: cohort table for a label
            concepts::list:: a list of concepts that will be used for train/infer
            label_name::str:: the name of label in this table
        out:
            folders,filenames,label,patch_nb:: 4 lists for creating dataloader
        """
        for name in concepts:
            df = df [df[name]>0]
        #df=self.cohort_df[[self.cohort_df[name]>0] for name in concepts]
        #df[name].values.tolist()
        folders = df['folder'].values.tolist()
        filenames = df["filename"].values.tolist()
        label = df[label_name].values.tolist()
        patch_nb = df["patch_nb"].values.tolist()
        label = [l[0] for l in label]
        slide_list = [[f,n,l] for f,n,l in zip(folders,filenames,label)]
        patch_list = [[f,n,i,l] for f,n,l,p_nb in zip(folders,filenames,label,patch_nb) for i in range(p_nb) ]
        logger.debug(f"Cohort::in data file list df:{df.keys()}, folders{len(folders)},patch nb = {len(patch_list)}")
        return slide_list,patch_list,patch_nb

    def get_concepts_nb(self,df,name:str,idx:int):
        nb_name = name+"_nb"
        return df[nb_name][idx]

    def is_in_table(self,df,folder,fname):
        state1 = (folder in df["folder"])
        state2 = (fname in df["filename"])
        return (state1 and state2)


class LocalCohort(Cohort):
    """
    Read local slides as a Cohort
    in::
        slide_root::pathlib.Path a folder path to slides files
    """
    def __init__(self,
                 slide_root:Path,
                 idx_root:Path,
                 cohort_name:str=None,
                 concepts:list=default_concept_keys,
                 concept_paras:CollectorParas=DEFAULT_CONCEPT_PARAS,
                 ):
        super().__init__(idx_root = idx_root,
                        concept_paras=concept_paras)
        """
        slide_root:Path: root (folder location) for all slide
        indx_root:Path:  root location for indexs csv
        concepts:list :  list for all related concepts in current cohort       
        """
        # data and idx loc
        self.slide_root = slide_root
        self.cohort_name = cohort_name
        self.concepts=concepts
        self.init_table(loc=self.loc())

    def loc(self,):
        return Path(str(self.idx_root) + f"/local_cohort_{self.cohort_name}.csv")

    def build(self,
                data_locs:Locations,
                pattern:str="*.svs"):
        """
        create a cohort for local slides
        in:
            data_locs:Locations: locations for data src
            model_name:str: name of feature extraction net
            pattern:str: format for local slide files
        """
        # create a df for all local slide
        self.local_slides(slide_root=self.slide_root,
                                        pattern=pattern,
                                        concepts=self.concepts)
        self.add_concept_to_df(concepts=self.concepts)
        # check concepts files and save to csv
        self.check_concepts(data_locs=data_locs,concepts=self.concepts)


    def usable_df(self,usbale_concepts:list= default_concept_keys):
        """
        get a dataframe with a list of concepts with saved data
        in:
            usbale_concepts:list: list of concepts that have data
        """
        df = self.table.df
        for name in usbale_concepts:
            df = df[df[name]==True]
        return df


class TaskCohort(Cohort):
    def __init__(self,
                #----> define task name
                task_name:str,
                #----> specify a task table
                idx_folder:Path,
                
                #----> for the item in cohort table
                cohort_paras:CohortParas,  #paras for this cohort
                task_concepts:list,
                concept_paras:CollectorParas=DEFAULT_CONCEPT_PARAS
                
                ) -> None:
        super().__init__(idx_root = idx_folder,
                        r_key=cohort_paras.r_key,
                        concept_paras=concept_paras)
        """
        Read a cohort from file input for a task
        in:
            task_name:str: name of task

            idx_folder:Path: root location for indexs  idx_root
            cohort_table:Path: file name for this cohort

            cohort_paras:CohortParas: paras for this cohort
            task_concepts:list: list of concepts that will be used for train/infer
            concept_paras:CollectorParas: paras for concept collector which can be used for train/infer

            cohort_file:str: loc of cohort file
            pid_name:str: name of patient id e.g. "PatientID"
            labels_name:list: list of labels e.g. ["label1","label2"]

        """
        self.cohort_paras = cohort_paras

        self.task_name = task_name
        self.concepts = task_concepts

        self.task_file = cohort_paras.cohort_file
        self.pid_name = cohort_paras.pid_name
        self.labels_name = cohort_paras.targets

        self.init_table(loc = self.loc())


    def loc(self):
        return Path(f"{str(self.idx_root)}/Task_{self.task_name}.csv")

    def save(self):
        self.table.update_loc(loc=self.loc())
        self.table.write_table()

    def build(self,local_df:pd.DataFrame):
        """
        build task cohort from source file
        """
        # get source table from a file
        source_table = tableWorker(loc = Path(f"{str(self.idx_root)}/{self.task_file}"))
        source_table.read_table()
        # merge source table with local table
        # add source hospital name
        content_names = self.cohort_paras.task_additional_idx+self.labels_name
        patinet_df = self.sort_patientID(table = source_table,pid_name=self.pid_name,labels_name=content_names)
        self.table.df = self.merge_with_PID( df_1=local_df,
                                            df_2=patinet_df)
        # write to file
        self.save()


    def get_task_datalist(self,dataframe:pd.DataFrame=None):
        """
        from a cohort select slides that have concept in concepts list, then build lists
        in:
            concepts:list,
            label_name:str
        """
        df = self.table.df if dataframe is None else dataframe
        return self.get_dataloader_lists(df=df,
                                        concepts=self.concepts,
                                        label_name=self.labels_name,
                                        )

    def get_concepts_stat(self,label_idx:str,concept_name:str="patch"):
        assert self.table.df is not None
        assert label_idx in self.labels_name
        df = self.table.df
        c_nb = df[concept_name+"_nb"].values.tolist()
        label = df[label_idx].values.tolist()
        all_cat = []
        for l_type  in set(label):
            c_nb_list = [nb if l==l_type else 0 for nb,l in zip(c_nb,label)]
            C_nb_np=np.array(c_nb_list)
            target_np=C_nb_np[C_nb_np.nonzero()]
            all_cat.append([l_type,target_np.tolist()])
        return all_cat






