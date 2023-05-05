"""
A simple wrapper for different backends

(1) openslide
(2) tifffiles

TO-DO:
# inspired by https://github.com/Dana-Farber-AIOS/pathml 
# there are other different formats that openslides not cover

# BioFormats parts depends on `python-bioformats <https://github.com/CellProfiler/python-bioformats>`_ 
# which wraps ome bioformats
(2) BioFormats
(3) DICOM files

"""
from HistoMIL import logger
import numpy as np
import os
from pathlib import Path
from urllib.parse import _NetlocResultMixinStr
from matplotlib.image import thumbnail
import openslide
import tifffile

class WSI_backends:
    """base class for backends that interface with slides on disk"""

    def _open_handler_(self):
        raise NotImplementedError

    def _close_handler_(self):
        raise NotImplementedError

    def get_region(self, coords, patch_size, patch_level, **kwargs):
        raise NotImplementedError

    def get_thumbnail(self, size, **kwargs):
        raise NotImplementedError

    def get_meta(self,):
        raise NotImplementedError

class WSI_Meta:
    def __init__(self,levels=None,level_dims=None,shape=None) -> None:
        # possible levels for wsi scale change
        # list [tuple:(scale_factor:int,int),tuple,..] default [(1.0,1.0)]
        self.levels = levels

        # possible downsamples 
        # list [tuple:(h:int,w:int),tuple,..] default[(w,h)]
        #self.downsamples = None

        # possible dimensions for different scales 
        # list [tuple:(h:int,w:int),tuple,..] default[(w,h)]        
        self.level_dims = level_dims
        # original size: Tuple(w:int,h:int)
        self.shape = shape

    def get_scale_factor(self,level_nb:int):
        assert (level_nb==-1 or (level_nb>=0 and level_nb <len(self.levels)))
        return self.levels[level_nb]

    def get_scaled_area(self,patch_size:tuple,level_nb:int):
        scale = self.get_scale_factor(level_nb=level_nb)
        return scale, int(patch_size**2 / (scale[0] * scale[1]))



def select_backends(loc:Path):
    format = loc.suffix
    #if format.lower() == ".dcm":

    if format in [".svs"]:
        return OpenSlideBackend(loc = str(loc))
    elif format in [".tiff",".tif"]:
        return TifffilesBackend(loc = str(loc))


###############################################################################
#       different backends for different formats
###############################################################################
class OpenSlideBackend(WSI_backends):
    def __init__(self,loc):
        logger.debug(f"FileIO:: Using openslide as backend.")
        self.loc = loc

        self.handler = None
        
    def _open_handler_(self):
        """
        open a WSI and set handler
        """
        try:
            self.handler = openslide.open_slide(filename=self.loc)
        except RuntimeError:
            raise RuntimeError(f"Cannot read wsi file {self.loc}")

    def _close_handler_(self):
        """
        close WSI file and set handler back to None
        """
        assert self.handler is not None
        self.handler.close()
        self.handler=None

    def get_region(self, coords:tuple, patch_size:tuple, patch_level:int):
        """
        read a small region of WSI
        coords: tuple, coords of the region's top left points
        patch_size: tuple, width and hight for region
        patch_level: the read level for this region
        """

        img = self.handler.read_region(coords, patch_level, patch_size).convert('RGB')
        img = np.array(img) # Convert to np array for processing
        return img

    def get_thumbnail(self, size:tuple):
        thumbnail = self.handler.get_thumbnail(size).convert('RGB').resize(size)
        thumbnail = np.array(thumbnail) # Convert to np array for processing
        return thumbnail

    def get_meta(self,):
        # get wsi levels 
        wsi_levels = [] #[(1.0, 1.0)]

        downsamples = self.handler.level_downsamples #[(img_w,img_h)]
        level_dims = self.handler.level_dimensions   #[(img_w,img_h)]
        dim_0 = level_dims[0]
        
        for d_sample, dim in zip(downsamples, level_dims):
            est_d_sample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))

            if est_d_sample != (d_sample, d_sample):
                wsi_levels.append(est_d_sample) 
            else:
                wsi_levels.append((d_sample, d_sample))

        # get img_w, img_h      
        img_w, img_h = self.handler.level_dimensions[0]

        return WSI_Meta(levels=wsi_levels,level_dims=level_dims,shape=(img_w, img_h ))




class TifffilesBackend(WSI_backends):
    def __init__(self,loc):
        logger.debug(f"FileIO:: Using tifffiles as backend.")
        self.loc = loc

        self.handler = None
        
    def _open_handler_(self):
        """
        open a WSI and set handler
        """
        assert (self.loc is not None and os.path.isfile(self.loc))
        try:
            self.handler = tifffile.imread(self.loc)
            #self.handler = np.asarray(data,dtype='float32')
            #self.handler = np.swapaxes(data,0,1)
            self.get_meta()
        except RuntimeError:
            raise RuntimeError(f"Cannot read wsi file {self.loc}")

    def _close_handler_(self):
        """
        close WSI file and set handler back to None
        """
        assert self.handler is not None
        #self.handler.close()
        self.handler=None

    def get_region(self, coords:tuple, patch_size:tuple, patch_level:int=None):
        """
        read a small region of WSI
        coords: tuple, coords of the region's top left points
        patch_size: tuple, width and hight for region
        patch_level: the read level for this region
        """
        slide = self.get_thumbnail(size=self.level_dims[patch_level])
        slide = np.swapaxes(slide,0,1)
        # only consider rgb channel
        img = slide[coords[0]:coords[0]+patch_size[0],coords[1]:coords[1]+patch_size[1],...]
        img = np.swapaxes(img,0,1)
        #img is np array
        return img


    def get_thumbnail(self, size:tuple):
        #get original shape
        # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
        o_size = self.handler.shape
        step_x = o_size[0]//size[0]
        step_y = o_size[1]//size[1]
        thumbnail = self.handler[::step_x,::step_y,...]
        thumbnail = thumbnail[:size[0],:size[1],...]
        return thumbnail

    def get_meta(self,):
        wsi_levels = [(1.0,1.0),(4.0,4.0),(16.0,16.0)]
        img_w = self.handler.shape[0]
        img_h = self.handler.shape[1]
        level_dims = [(img_w,img_h),
                        (int(img_w//4.0),int(img_h//4.0)),
                        (int(img_w//16.0),int(img_h//16.0)),]
        self.level_dims = level_dims
        return WSI_Meta(levels=wsi_levels,level_dims=level_dims,shape=(img_w, img_h ))



class DICOMBackend(WSI_backends):
    def __init__(self,loc):
        logger.debug(f"FileIO:: Using wsidicom package as backend.")
        self.loc = loc

        self.handler = None
        
    def _open_handler_(self):
        """
        open a WSI and set handler
        """
        pass

    def _close_handler_(self):
        """
        close WSI file and set handler back to None
        """
        pass

    def get_region(self, coords:tuple, patch_size:tuple, patch_level:int=None):
        """
        read a small region of WSI
        coords: tuple, coords of the region's top left points
        patch_size: tuple, width and hight for region
        patch_level: the read level for this region
        """
        pass


    def get_thumbnail(self, size:tuple):
        pass





        


