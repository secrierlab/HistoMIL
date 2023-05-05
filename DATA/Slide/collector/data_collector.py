"""
Integrate different base formats

i.e. integrate whole slide image, related masks, and related meta_data together
"""
from opcode import hasconst
import os
import sys
from HistoMIL import logger
from pathlib import Path

from HistoMIL.DATA.Slide.concepts.WholeSlideImage import WholeSlideImage
from HistoMIL.DATA.Slide.concepts.feature import Features
from HistoMIL.DATA.Slide.concepts.patch import Patches
from HistoMIL.DATA.Slide.concepts.tissue import TissueMask

sys.path.append("..") 
sys.path.append(".") 
import random
from typing import Iterable, Optional, Tuple, Union
import pathlib
import numpy as np
##############################################################################
#             import from utils
##############################################################################
from HistoMIL.DATA.Slide.collector.items import DataCollector
from HistoMIL.DATA.Cohort.location import Locations

from HistoMIL.EXP.paras.slides import DEFAULT_CONCEPT_PARAS,CollectorParas

DEFAULT_CONCEPT_DICT = {"slide":WholeSlideImage,
                        "tissue":TissueMask,
                        "patch":Patches,
                        "feature":Features,
                  }

# default possible concepts for a cohort
default_concept_keys = list(DEFAULT_CONCEPT_DICT.keys())

##############################################################################
#             data box class main code
##############################################################################
DEFAULT_CONCEPT_TYPE = DEFAULT_CONCEPT_DICT.keys()

class WSICollector(DataCollector):
    
    def __init__(self,db_loc:Locations,
                      wsi_loc:str,
                      paras:CollectorParas = DEFAULT_CONCEPT_PARAS) -> None:
        """
        A collector for all possible concepts processing include segmentation,
        patching,feature extraction, 
        in:
            db_loc:Locations,  # location for databases include different
                               # folders for different concepts
            wsi_loc:Path # location of a whole slide image
            paras:dict = DEFAULT_CONCEPT_PARAS # paras to get different concepts
        
        out: None

        example:
        collector = WSICollector(exp.data_locs,wsi_loc)
        collector.create("wsi")
        collector.read()
        """

        self.default_concepts=DEFAULT_CONCEPT_DICT # link to construction functions

        self.db_loc = db_loc
        self.wsi_loc = wsi_loc

        self.paras = paras

        self.slide:WholeSlideImage = None
        self.tissue:TissueMask = None # tissue mask can be counter or mask image
        self.patch:Patches = None # patch detials 
        self.feature:Features = None # feature instances
    ############################################################
    #     dictionary related functions
    ############################################################
    def is_concept(self,name:str)->bool:
        """
        check whether name is supported by current concept list
        """
        return name in self.default_concepts
    ############################################################
    #     concept instance related functions
    ############################################################

    def create(self,name:str) -> str:
        """
        create an instance for a concept_name
        in:
            concept_name:str: name of the concept
            paras:object: paras for the concept
        out:
            instance_name:str: name of the instance
        """
        logger.debug(f"Collector:: Creating instance for {name}")

        instance = self.default_concepts[name](db_loc=self.db_loc,
                                                   wsi_loc=self.wsi_loc,
                                                   paras=self.paras.__getattribute__(name))
        
        self.__setattr__(name,instance)
        logger.debug(f"Collector:: Created instance for {name}: {self.__getattribute__(name)}")
        return name


    def read(self,name:str):
        logger.debug(f"Collector:: Reading data for instance {name}.")
        self.__getattribute__(name).read()

    def get(self,name:str,force_calc:bool=False):
        """
        An all in on function to create an instance and read/calc from files
        this function is used to simplify the following functions
        in:
            name:str: name of the concept
            idx:str: index of the concept set as "default" if only one instance
            req_idx_0,req_idx_1,req_idx_2:str: index of the concept that is pre-request for calculate target concept instance
        """
        # initialis an concept instance
        if not hasattr(self,name): 
            self.create(name)
        
        logger.debug(f"Collector:: Get data for instance {name}")

        if name == "slide":
            self.slide.get()
        elif name == "tissue":
            self.tissue.get(self.slide,force_calc=force_calc)
        elif name == "patch":
            self.patch.get(self.slide,self.tissue,force_calc=force_calc)
        elif name == "feature":
            self.feature.get(self.slide,self.patch,force_calc=force_calc)

    def release(self,name:str,idx:str="default"):
        self.__delattr__(name)
    
    ############################################################
    #     instance data related functions
    ############################################################

    def with_saved_data(self,name):
        """
        check whether concept(name) have saved data
        in:
            name:str: name of the target concept
        out:
            out_flag:bool: have file and can read it, return True else return false
            out_nb:int: only if name=="patch", return number of patches
        """
        out_flag = None
        out_nb   = -1

        try:
            self.create(name)
            self.read(name)
        except Exception as e:
            logger.info(f"Collector::Fail to read {name}.")
            out_flag = False
            out_nb =-1
        else:
            out_flag = True
            out_nb = self.__getattribute__(name).len()
        finally:
            self.release(name)
        return out_flag,out_nb

    def item_loc(self,name:str)->str:
        if self.__getattribute__(name)is None:
            logger.debug(f"Collector::Not find {name}, creating a new instance.")
            self.create(name)
        loc = self.__getattribute__(name).loc()
        logger.debug(f"Collector:: Item {name} locate at {loc}")
        return loc

    def get_patch_data(self,name:str,patch_idx:int)->tuple:
        """
        get data for a patch with patch_idx
        in:
            name:str: name of the concept
            idx:str: index of the concept set as "default" if only one instance
            patch_idx:int: index of the patch in a slide
        out:
            coords:tuple: (x,y) of the patch
            data:np.ndarray: data for the patch
        """
        assert name in ["patch","feature"]
        assert hasattr(self,name)

        if name == "patch":
            # get a patch image (np.array)
            coords,data = self.patch.get_datapoint(slide=self.__getattribute__("slide"),
                                                                    idx=patch_idx)
        elif name == "feature":
            # get a feature vector for a patch
            # for feature name should be f"{self_idx}_{slide_idx}_{patch_idx}" 
            if hasattr(self,"patch"):
                coords = self.patch.coords_dict[patch_idx]
            else:
                logger.info("Collector:: Warning: can not read from patch instance, coords set to []")
                coords = []
            data = self.feature.get_datapoint(idx=patch_idx)

        return coords,data
       


def read_wsi_collector(data_locs,
                        folder:str, 
                        fname:str,
                        concepts:list,
                        paras:CollectorParas=DEFAULT_CONCEPT_PARAS):
    C = WSICollector(db_loc=data_locs,
                    wsi_loc=folder+"/"+fname,
                    paras = paras)

    for name in concepts:
        # feature need to specify model_name
        C.create(name)
        C.read(name)
    return C

def pre_process_wsi_collector(data_locs,
                            wsi_loc:Path,
                            collector_paras:CollectorParas,
                            concepts:list=["slide","tissue","patch"],
                            fast_process:bool=True,force_calc:bool=False):

    C = WSICollector(db_loc=data_locs,wsi_loc=wsi_loc,paras=collector_paras)
    try:

        for name in concepts:
            if name == "tissue":
                if fast_process:
                    from HistoMIL.EXP.paras.slides import set_min_seg_level
                    C.paras.tissue = set_min_seg_level(C.paras.tissue, C.slide,C.paras.tissue.min_seg_level)
                    logger.debug(f"Collector:: set seg level to {C.paras.tissue.seg_level}")
            C.create(name)
            C.get(name, force_calc) # for tissue, req_idx_0 is always default slide
    except Exception as e:
        logger.exception(e)
    else:
        logger.info(f"Collector:: {wsi_loc} is done")
    finally:
        del C

