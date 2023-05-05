
import attr 

@attr.s(auto_attribs=True)
class DSMILParas:
    """
    include all paras for DSMIL model
    """
    #------> parameters for model
    encoder_name="pre-calculated"
    encoder_pretrained:bool = True # or False

    feature_dim:int = 1024
    p_class:int = 2 # number of classes for the prediction of the patch
    b_class:int = 2 # number of classes for the prediction of the bag
    dropout_r:float = 0.5