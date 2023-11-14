"""
pre-defined parameters for TransMIL model
"""
import torch.nn as nn
import attr 

#---->

@attr.s(auto_attribs=True)
class TransMILParas:
    """
    include all paras for create TransMIL model
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False
    feature_size:int=512
    embed_size:int=None

    n_classes:int=2
    norm_layer=nn.LayerNorm
    #class_nb:int=1

    #------> parameters for feature encoder
    #backbone:str="pre-calculated"
    #pretrained:bool=True
