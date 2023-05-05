"""
Generate patch for a WSI and read/write patch info from file 
"""
from PIL import Image

import multiprocessing as mp
import math
import os
import cv2
import numpy as np
import h5py
import sys 
sys.path.append("..") 
sys.path.append(".") 
from pathlib import Path
##############################################################################
#             import from other files
###############################################################################
from HistoMIL import logger
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Slide.collector.items  import Items

from HistoMIL.DATA.Slide.concepts.WholeSlideImage import WholeSlideImage
from HistoMIL.DATA.Slide.concepts.tissue import TissueMask
from HistoMIL.DATA.Slide.concepts.utils import isInContours,set_contour_check_fn

from HistoMIL.DATA.FileIO.h5_worker import patchCoordsStorage

from HistoMIL.EXP.paras.slides import PatchParas
##############################################################################
#             main code
###############################################################################
class Patches(Items):
    def __init__(self,db_loc:Locations,wsi_loc:str,paras:PatchParas) -> None:
        logger.debug("Patch:: Init instance for Patches")
        self.db_loc = db_loc
        self.paras = paras
        self.wsi_loc = Path(wsi_loc)
        self.patch_loc = self.loc()

        self.coords_dict = None
        self.attr_dict = None


    def pre_requests(self):
        return ["slide","tissue"]

    def loc(self):
        folder = str(self.wsi_loc.parent).split("/")[-1]
        fname  = self.wsi_loc.name
        return Path(self.db_loc.abs_loc("patch")+f"/{self.paras.patch_size[0]}_{self.paras.step_size}/{folder}.{fname}.h5")

    def calc(self,slide:WholeSlideImage,tissue:TissueMask,paras:PatchParas):
        if paras.from_contours:
            self.calc_with_contours(slide=slide,tissue=tissue,paras=paras)
        else:
            self.calc_with_mask(slide=slide,tissue=tissue,paras=paras)
    
    def read(self,loc:str=None):
        if loc is None:
            loc = self.patch_loc
        h5worker = patchCoordsStorage(loc = loc)
        self.attr_dict = h5worker.read('coords',is_attrs=True,
                                                attrs=["patch_level","patch_size"])
        self.coords_dict = h5worker.read('coords',is_attrs=False)

        self.patch_level = self.attr_dict["patch_level"]
        self.patch_size  = self.attr_dict["patch_size"] #tuple (h,w)
        #self.step_size   = self.attr_dict["step_size"]
        logger.debug(f"Patch:: read {len(self.coords_dict)} patches (level:{self.patch_level},{self.patch_size})from file.")
    
    def _save_step(self,is_init:bool,h5worker:patchCoordsStorage,coord_dict, attr_dict= None):

        if is_init:
            h5worker.save(coord_dict, attr_dict, mode='w')
            is_init = False
        else:
            h5worker.save(coord_dict, mode='a')
        
        return is_init
    
    def len(self):
        return len(self.coords_dict)

    def get(self,slide:WholeSlideImage,tissue:TissueMask,
                paras:PatchParas=None,force_calc:bool=False):
        self.paras = paras if paras is not None else self.paras
        #req_dict
        if os.path.isfile(self.patch_loc) and not force_calc:
            self.read()
        else:
            self.slide = slide
            self.tissue = tissue
            self.calc(slide = slide,
                      tissue = tissue,
                      paras=self.paras)        
    
    def get_datapoint(self,slide:WholeSlideImage,idx:int):
        """
        get an image patch with idx
        in:
            slide::WholeSlideImage:: original slide
            idx:int: idx of the patch 
        out:
            img::np.array:: patch numpy array with (H x W x C)
        """
        slide.read()
        coords = self.coords_dict[idx]
        img=slide.get_region(coords = coords,
                             patch_size = self.patch_size,
                            patch_level = self.patch_level)
        slide.close()
        return coords,img
    ##############################################################################
    #             calc patching H5
    ###############################################################################
    
    def _assert_patch_start(self,contour):
        # set start point
        assert self.patch_level is not None

        default_start = (0, 0,
                        self.slide.meta.level_dims[self.patch_level][0],
                        self.slide.meta.level_dims[self.patch_level][1])
        start_x, start_y, w, h = cv2.boundingRect(contour) if contour is not None else default_start

        return start_x, start_y, w, h

    ################### Internal utils  functions to calc patching ###########################
    def _calc_patch_downsample_factors(self):
        # for the selected patch_level, calc actural patch size for wsi
        patch_downsample = (
                            int(self.slide.meta.levels[self.patch_level][0]),
                            int(self.slide.meta.levels[self.patch_level][1])
                            )
        ref_patch_size = (
                          self.patch_size[0]*patch_downsample[0],
                          self.patch_size[1]*patch_downsample[1]
                          )
        return patch_downsample,ref_patch_size
        
    def _set_Patching_padding(self,use_padding,
                                   start_x,start_y,w,h,
                                   img_h,img_w,ref_patch_size): 
        # set padding for patching
        if use_padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
        return stop_x,stop_y

    def _set_Patching_area(self,bot_right,top_left,start_x,start_y,stop_x,stop_y):
        # set start point and stop point for patching
        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                #logger.debug("Patch:: Contour is not in specified ROI, skip")
                pass
            else:
                #logger.debug("Patch:: Adjusted Bounding Box:", start_x, start_y, w, h)
                pass
        return start_x,start_y,stop_x,stop_y

    def _calc_Patching_coord_candidates(self,start_x,start_y,stop_x,stop_y,
                                            step_size,patch_downsample):
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        # use numpy to get coords candicate list
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()
        return coord_candidates

    def _calc_patch_mp(self,step_coord_InContour,iterable):
        # multiple processing function
        num_workers = mp.cpu_count()
        us_workers = self.paras.mp_processor 
        if us_workers is None:
            if num_workers > 32: us_workers = 4
            else:us_workers = 1
        logger.debug(f"Patch:: Use num_workers {us_workers}/{mp.cpu_count()}")
        pool = mp.Pool(us_workers)

        #iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(step_coord_InContour, iterable)
        pool.close()
        results = np.array([result for result in results if result is not None])
        return results

    ################### Actural internal function to calc patching ###########################
    def _calc_patches(self, 
                    contour, contour_holes, # seg results
                    # with kwargs
                    paras:PatchParas): #specify patching area
        """
        get patches for one contour
        In: contour -> Out: patch_coord_list

        """
        # read original w and h for patching
        (img_w, img_h) = self.slide.meta.shape
        

        # set start point
        start_x, start_y, w, h = self._assert_patch_start(contour)

        # set patches para for downsample
        patch_downsample,ref_patch_size = self._calc_patch_downsample_factors()

        # padding selection
        stop_x,stop_y = self._set_Patching_padding(paras.use_padding,
                                                        start_x,start_y,
                                                        w,h,
                                                        img_h,img_w,
                                                        ref_patch_size)
        
        #print("Bounding Box:", start_x, start_y, w, h)
        #print("Contour Area:", cv2.contourArea(contour))

        # set start point and stop point
        start_x,start_y,stop_x,stop_y = self._set_Patching_area(paras.bot_right,
                                                                paras.top_left,
                                                                start_x,
                                                                start_y,
                                                                stop_x,
                                                                stop_y)
        
        if paras.bot_right is not None or paras.top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                logger.debug("Patch:: Contour is not in specified ROI, skip")
                return {},{}


        # select contour check fn
        cont_check_fn = set_contour_check_fn(paras.contour_fn_name,contour,ref_patch_size)
        
        # generate coord_candidates
        coord_candidates = self._calc_Patching_coord_candidates(start_x,start_y,
                                                                stop_x,stop_y,
                                                                self.step_size,patch_downsample)

        # multiple processing for is in the contour calculation
        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = self._calc_patch_mp(step_coord_InContour,iterable)

        logger.debug('Patch:: Extracted {} coordinates'.format(len(results)))

        if len(results)>1:
            coords_dict = {'coords' :          results}
            
            attr = {
                    'patch_size' :            self.patch_size, # To be considered...
                    'patch_level' :           self.patch_level,
                    'step_size'  :            self.step_size,
                    'downsample':             self.slide.meta.levels[self.patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.slide.meta.level_dims[self.patch_level])),
                    'name':                   str(self.slide.wsi_loc),
                    }

            attr_dict = { 'coords' : attr}
            return coords_dict, attr_dict

        else:
            return {}, {}
    def calc_with_mask(self,slide:WholeSlideImage,tissue:TissueMask,paras:PatchParas):
        # calc patches with mask
        logger.debug("Patch:: calculating patches")
        self.slide = slide
        self.tissue = tissue
        self.patch_level = paras.patch_level#["patch_level"]
        self.patch_size  = paras.patch_size#["patch_size"]
        self.step_size   = paras.step_size#["step_size"]
        self.slide.read()
        scale_factor = self.slide.meta.get_scale_factor(self.patch_level)
        stop_x= int(self.slide.meta.shape[0]//scale_factor[0])
        stop_y= int(self.slide.meta.shape[1]//scale_factor[1])
        p_x = int(self.patch_size[0]//scale_factor[0])
        p_y = int(self.patch_size[1]//scale_factor[1])
        assert self.tissue.mask_img is not None
        mask = self.tissue.mask_img.copy()
        mask =  np.swapaxes(mask,0,1)
        mask_img = Image.fromarray(mask).resize((stop_y,stop_x))
        resized = np.asarray(mask_img)

        # set patches para for downsample
        patch_downsample,ref_patch_size = self._calc_patch_downsample_factors()

        x_range = np.arange(0, stop_x, step=int(self.step_size//scale_factor[0]))
        y_range = np.arange(0, stop_y, step=int(self.step_size//scale_factor[1]))
        # use numpy to get coords candicate list
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        #print(f"{len(coord_candidates)}")
        results = []
        for coor in coord_candidates:
            x = coor[0] #+ int(p_x/2)
            y = coor[1] #+ int(p_y/2)
            #print(f"{resized.shape}: : {x+p_x}/ {stop_x}, {y+p_y}/{stop_y} : {coor} ")
            if x+p_x<stop_x and y+p_y<stop_y:
                #con1 = (resized[x,y]!=0 and resized[x+ int(p_x/2),y+ int(p_y/2)]!=0)
                #con2 = (resized[x+p_x,y+p_y]!=0 and resized[x+ int(p_x/2),y+ int(p_y/2)]!=0)
                if is_in_mask(resized,x,y,p_x,p_y,shift=0.3):
                    results.append(coor)
        scaled_results = [(int(coord[0]*scale_factor[0]),int(coord[1]*scale_factor[1])) for coord in results]

        results = np.asarray(scaled_results)
        if len(results)>1:
            coord_dict = {'coords' :          results}
            
            attr = {
                    'patch_size' :            self.patch_size, # To be considered...
                    'patch_level' :           0,#self.patch_level,
                    'step_size'  :            self.step_size,
                    'downsample':             self.slide.meta.levels[self.patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.slide.meta.level_dims[self.patch_level])),
                    'name':                   str(self.slide.wsi_loc),
                    }

            attr_dict = { 'coords' : attr}
        is_init = True
        h5worker = patchCoordsStorage(loc = self.patch_loc)
        self._save_step(is_init,h5worker, coord_dict, attr_dict)
        logger.debug("Patch:: Patching H5 Done..")
        self.slide.close()
        self.read()    
        
    def calc_with_contours(self,slide:WholeSlideImage,tissue:TissueMask,paras:PatchParas):
        logger.debug("Patch:: calculating patches")
        self.slide = slide
        self.tissue = tissue
        self.patch_level = paras.patch_level#["patch_level"]
        self.patch_size  = paras.patch_size#["patch_size"]
        self.step_size   = paras.step_size#["step_size"]

        if self.tissue.contours_tissue is None or len(self.tissue.contours_tissue)<1:
            logger.debug("Patch:: No contour tissue is recognised..")
            return None
        # Start processing
        n_contours = len(self.tissue.contours_tissue)

        logger.debug(f"Patch:: Total number of contours to process: {n_contours}")
        # set some settings
        fp_chunk_size = math.ceil(n_contours * 0.05)
        is_init = True

        h5worker = patchCoordsStorage(loc = self.patch_loc)
        self.slide.read()
        # start processing..
        for idx, cont in enumerate(self.tissue.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                logger.debug(f'Patch:: Processing contour {idx}/{n_contours}')

            # process a contour
            coord_dict, attr_dict = self._calc_patches(contour=cont,
                                                    contour_holes=self.tissue.holes_tissue[idx],
                                                     paras=paras)
            if len(coord_dict) > 0:
                #print(coord_dict, attr_dict)
                is_init = self._save_step(is_init,h5worker, coord_dict, attr_dict)
        logger.debug("Patch:: Patching H5 Done..")
        self.slide.close()
        self.read()     


def step_coord_InContour(coord, contour_holes, ref_patch_size, cont_check_fn):
    if isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
        return coord
    else:
        return None

def is_in_mask(mask,x,y,p_x,p_y,shift=0.3):
    center_x = x + int(p_x/2)
    center_y = y + int(p_y/2)
    shift_x = int((p_x/2)*shift)
    shift_y = int((p_y/2)*shift)
    pts = [
        [center_x,center_y],
        [center_x+shift_x,center_y+shift_y],
        [center_x-shift_x,center_y-shift_y],
        [center_x+shift_x,center_y-shift_y],
        [center_x-shift_x,center_y+shift_y],
    ]
    flag = True
    for [x,y] in pts:
        if mask[x,y]==0:
            flag=False
    return flag
"""

"""