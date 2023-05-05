"""
A basic component for different pixel-wised mask labels

Different data source may have different way to label data
A widely considered label is pixel-wised mask

"""
import numpy as np
import cv2
import os

from pathlib import Path
#############################################################################
#               import within package
#############################################################################
from HistoMIL import logger
from HistoMIL.DATA.FileIO.pkl_worker import save_pkl,load_pkl
from HistoMIL.DATA.Slide.concepts.utils import scaleContourDim,scaleHolesDim,filter_contours 

# related concepts
from HistoMIL.DATA.Slide.concepts.WholeSlideImage import WholeSlideImage

# func to calc scaled seg results
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Slide.collector.items  import Items

# pre-defined paras
from HistoMIL.EXP.paras.slides import TissueParas


#############################################################################
#               Main Source
#############################################################################
class TissueMask(Items):
    def __init__(self,db_loc:Locations,wsi_loc:str,paras:TissueParas) -> None:
        """
        Object wrapping for pixel-wised segmentation labels 
        Args:
            name: the name of this label
            mask: the pixel-wised label
        """
        logger.debug("Tissue:: Init instance for Tissue Mask")

        self.db_loc = db_loc
        self.wsi_loc = Path(wsi_loc)
        self.paras = paras
        self.tissue_loc = self.loc()

        self.seg_level= None

        # masks as image
        self.mask_img = None 
        # masks as contours
        self.holes_tissue = None
        self.contours_tissue = None

    def pre_requests(self):
        return ["slide"]

    def loc(self):
        folder = str(self.wsi_loc.parent).split("/")[-1]
        fname  = self.wsi_loc.name
        return self.db_loc.abs_loc("tissue")+folder+"."+fname+".pkl"

    def calc(self,slide:WholeSlideImage,paras:TissueParas):
        logger.debug(f"Tissue::Calc tissue ..")
        img_otsu = self.get_mask(slide=slide,paras=paras)      
        #self.mask_img = np.swapaxes(img_otsu,0,1)    
        self.mask_img = img_otsu
        # Find and filter contours
        if paras.to_contours:
            self.mask2contours(slide,paras)
    
    def read(self):
        loc = self.tissue_loc
        logger.debug(f"Tissue::Read tissue from file{loc}")

        seg_dict = load_pkl(filename=loc)
        if "tissue" in seg_dict.keys():
            self.holes_tissue = seg_dict['holes']
            self.contours_tissue =seg_dict['tissue']
        if "mask" in seg_dict.keys():
            self.mask_img = seg_dict["mask"]
                
            
    def len(self):
        return 1

    def save(self):
        loc = self.tissue_loc
        logger.debug(f"Tissue::Save tissue to file{loc}")
        # save segmentation results using pickle
        if self.mask_img is not None and self.contours_tissue is None:
            asset_dict = {"mask":self.mask_img}
        elif self.contours_tissue is not None:
            asset_dict = {'holes': self.holes_tissue, 'tissue': self.contours_tissue}
        else:
            raise ValueError("No tissue mask or contours to save")
        save_pkl(loc, asset_dict)
        logger.debug(f"Tissue::Seg pkl {loc} saved..")   

    def get(self,slide:WholeSlideImage,paras:TissueParas=None,force_calc:bool=False):
        self.paras = paras if paras is not None else self.paras
        #req_dict
        if os.path.isfile(self.tissue_loc) and not force_calc:
            self.read()
        else:
            self.calc(slide=slide,paras=self.paras)
            self.save()

    #############################################################################
    #           calculate transfor between mask and counters
    #############################################################################
    def get_slide_img(self,data:WholeSlideImage,seg_level):
        # wsi original data related part
        if data.worker.handler is None:
            data.read()
        if data.meta.levels is None:
            data.get_meta()
        
        # read from wsi with selected seg_level use openslide
        size=data.meta.level_dims[seg_level]
        wsi_img = data.get_thumbnail(size=size)
        data.close()
        return wsi_img
    # internal functions for segmentation calc
    def mask2contours(self,
                     slide:WholeSlideImage,
                     paras:TissueParas):
        """
        calculate contours 
        in: img_otsu # original mask candidate
            keep_ids # whether keep those
            exclude_ids # whehter exclude
        """
        logger.debug("Tissue::transfer mask to contours for further processing.")
        # get paras
        
        patch_size = paras.ref_patch_size #["ref_patch_size"]
        filter_params = paras.filter_params #["filter_params"].copy()
        # calc actual patch size
        scale,scaled_patch_area = slide.meta.get_scaled_area(patch_size,
                                                            level_nb=self.seg_level)
        
        filter_params['a_t'] = filter_params['a_t'] * scaled_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_patch_area
        # calc original contours with scale
        contours, hierarchy = cv2.findContours(self.mask_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params: 
            # Necessary for filtering out artifacts
            foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params) 

        contours_tissue = scaleContourDim(foreground_contours, scale)
        holes_tissue = scaleHolesDim(hole_contours, scale)       

        # use keep_ids
        if len(paras.keep_ids) > 0:
            contour_ids = set(paras.keep_ids) - set(paras.exclude_ids)
        else:
            contour_ids = set(np.arange(len(contours_tissue))) - set(paras.exclude_ids)
        
        # direct set seg out item
        self.contours_tissue = [contours_tissue[i] for i in contour_ids]
        self.holes_tissue = [holes_tissue[i] for i in contour_ids]
        logger.debug(f"Tissue::Found contours_tissue: {len(self.contours_tissue)}")

    def get_mask(self, slide:WholeSlideImage,paras:TissueParas):
        self.seg_level = paras.seg_level #["seg_level"]
        img = self.get_slide_img(slide,self.seg_level)

        # convert to HSV space and apply blurring
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], paras.mthresh)  # Apply median blurring ["mthresh"]
        
        
        # Thresholding
        if paras.use_otsu:
            _, img_otsu = cv2.threshold(img_med, 
                                        0, 
                                        paras.sthresh_up, 
                                        cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, 
                                        paras.sthresh, 
                                        paras.sthresh_up, 
                                        cv2.THRESH_BINARY)

        # Morphological closing
        if paras.close > 0:
            kernel = np.ones((paras.close, paras.close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)   
        return  img_otsu


