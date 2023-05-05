"""
parameters for attention-based multiple instance learning

include pre-defined parameters to make the model runable
"""


import attr 

@attr.s(auto_attribs=True)
class AttMILParas:
    """
    include all paras for AttMIL model
    """
    #------> parameters for model
    encoder_name="pre-calculated"# or name of encoder",
    encoder_pretrained:bool = True # or False

    feature_dim:int = 1024
    hidden_dim:int=512
    class_nb:int=1

    sample_nb:int = 200
    #------> parameters for loss and optimizer
    loss_scale_att:float = 0.5
    loss_scale_feat:float = 0.5

    #------> parameters for pl protocol
    update_feature_extractor:bool = False
    max_instances:int = 48 # max number of instances in a small batch within bag