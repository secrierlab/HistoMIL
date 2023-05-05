"""
Related paras for concepts collector
mainly used for pre-processing
"""
from typing import Callable
import attr
from HistoMIL import logger
from HistoMIL.DATA.Slide.concepts.WholeSlideImage import WholeSlideImage 

from HistoMIL.DATA.Database.data_aug import naive_transforms,only_naive_transforms


##############################################################################
#           para for slide
##############################################################################

@attr.s(auto_attribs=True)
class SlideParas(object):
    
    folder:str=None
    fname:str = None

##############################################################################
#           para for tissue
##############################################################################
@attr.s(auto_attribs=True)
class TissueParas(object):
    """
    include all paras for tissue concepts in pre-processing and usage
    """
    seg_level:int = 0 # level for segment tissue mask
    min_seg_level:int = None # min level for segment tissue mask if chose fast mode

    ref_patch_size:int = 256 # reference patch size for tissue mask

    #------> parameters for blurring
    mthresh:int = 7 # paras for Apply median blurring

    #------> parameters for otsu
    use_otsu:bool = True
    sthresh:int = 20   
    sthresh_up:int = 255

    
    #------> Morphological closing
    close:int = 0

    #------> parameters for contours in mask2contours()
    filter_params:dict = {'a_t':100,'a_h': 16, 'max_n_holes':8}

    # if there is more than one contours, exclude option:default empty list
    to_contours:bool = True
    exclude_ids:list = []
    keep_ids:list = []
    
    #------> create a name for instance
    name:str = f"tissue_{seg_level}_otsu_{use_otsu}_contours_{to_contours}"

def set_min_seg_level(tissue_para:TissueParas,slide:WholeSlideImage,
                        min_seg_level:int=None):
    """
    get minimum seg level for tissue mask
    """
    if min_seg_level is None:
        tissue_para.seg_level = len(slide.meta.level_dims)-1
    else:
        tissue_para.seg_level = min(len(slide.meta.level_dims)-1,min_seg_level)
    logger.info(f"TissuePara:: set min_seg_level to {tissue_para.seg_level},in {slide.meta.level_dims} ")
    return tissue_para

##############################################################################
#           para for patch
##############################################################################
@attr.s(auto_attribs=True)
class PatchParas(object):
    """
    include all paras for patch concepts in pre-processing and usage
    """
    #------> parameters for patch
    patch_level:int = 0 # level for patch
    patch_size = (512,512) # patch size
    step_size:int = 512 # step size for patch

    #------> parameters for patch extraction
    from_contours:bool = True # extract patches from contours otherwise from tissue mask
    # debug: set mp to 1 to avoid not solved error 
    mp_processor:int = 1 # number of processors for multiprocessing
    #------> parameters for patch extraction function 
    contour_fn_name:str = "four_pt" # function name for contour extraction
    use_padding:bool = True      # whether padding
    top_left = None # top left point for patch extraction area
    bot_right = None # bot right point for patch extraction area

    #------> name for instance
    name:str = f"patch({patch_level})_size({patch_size[0]})_step({step_size})_contours({contour_fn_name})"


##############################################################################
#           para for faeture
##############################################################################
@attr.s(auto_attribs=True)
class FeatureParas(object):
    """
    include all paras for feature concepts in pre-processing and usage
    """
    #------> parameters for feature encoder
    model_name:str = "resnet18"

    model_instance = None
    img_size = None
    out_dim = None
    #-----> for inference part 

    device:str = "cuda"
    trans:Callable = only_naive_transforms
    
    batch_size:int = 32

    #------> parameters for cluster
    cluster_nb:int = 200
    with_semantic_shifts:bool = False

##############################################################################
#           para for collectorß
##############################################################################
@attr.s(auto_attribs=True)
class CollectorParas(object):
    """
    include all paras for collector concepts in pre-processing and usage
    """
    #------> parameters for collector
    slide:SlideParas = SlideParas() # get instance of slide paras
    tissue:TissueParas = TissueParas() # get instance of tissue paras
    patch:PatchParas = PatchParas() # get instance of patch paras
    feature:FeatureParas = FeatureParas()

DEFAULT_CONCEPT_PARAS = CollectorParas()

