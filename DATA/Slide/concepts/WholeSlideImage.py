"""
This file is use to define WSI i/o related class for the project

(1) WSI IO with svs format

"""
##############################################################################
#             related import
###############################################################################
import sys 
from PIL import Image

from pathlib import Path

Image.MAX_IMAGE_PIXELS = 933120000
sys.path.append("..") 
sys.path.append(".") 
##############################################################################
#             import from other files
###############################################################################
from HistoMIL import logger
from HistoMIL.DATA.FileIO.wsi_backends import select_backends
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Slide.collector.items  import Items
#############################################################################
#              Whole Slide Image Source
#############################################################################
class WholeSlideImage(Items):
    def __init__(self,db_loc:Locations,wsi_loc:str,paras=None) -> None:
        logger.debug("Init instance for Whole Slide Image.")
        self.db_loc = db_loc
        self.paras = paras
        self.wsi_loc = Path(str(self.db_loc.abs_loc("slide"))+"/" + str(wsi_loc))
        self.worker = select_backends(self.wsi_loc)
        #self.get_meta()

    def set_backends(self,loc:str=None):
        actural_loc = Path(loc) if loc is not None else Path(self.wsi_loc)
        self.worker = select_backends(actural_loc)

    def pre_requests(self):
        return []

    def get(self):
        self.read()
    
    def loc(self):
        return self.wsi_loc

    def read(self):
        #logger.debug(f"WSI::Initialise file handler for {self.loc()}.")
        self.worker._open_handler_()
        self.get_meta()

    def len(self):
        return 1

    def close(self):
        #logger.debug(f"WSI::Remove file handler for {self.loc()}.")
        self.worker._close_handler_()

    def get_region(self, coords, patch_size, patch_level,**kwargs):
        #logger.debug(f"WSI::get region at {coords} with patch size {patch_size} at level {patch_level}.")
        img = self.worker.get_region(coords, patch_size, patch_level,**kwargs)
        return img

    def get_thumbnail(self, size, **kwargs):
        logger.debug(f"WSI::get thumbnail with size {size}.")
        return self.worker.get_thumbnail(size,**kwargs)

    def get_meta(self,):
        self.meta = self.worker.get_meta()




