import attr
@attr.s(auto_attribs=True)
class PLTrainerParas(object):
    """
    include all paras for pl trainer
    """
    #------> parameters for model
    method_type:str = "mil" # "mil" or "patch_learning"
    model_name:str = "TransMIL"       # "nuclei" or "patch"
    model_para=None               # paras for create basic torch model

    backbone_name:str = "resnet18" # "resnet18" or "resnet50"
    model_out_list:list = ["logits","Y_prob","Y_hat",] # check needed model output
    
    #------> parameters for pl protocol
    task_type:str = "classification" # "classification" or "regression"
    
    metric_names:list = ['auroc','accuracy'] # should be lower case, 
    #full list can be found in histocore/MODEL/Image/PL_protocol/MetricsFactory.py
    
    label_format:str = "int"  # or "one_hot" for classification or "float" regression or "box" or "mask" for detection.segmentation

    shuffle_data:bool = True
    k_fold:int = 4 # None for no k-fold
    #------> parameters for pl trainer
    out_loc:str = None
    with_logger:str = "wandb"

    #------> parameters for ckpt
    with_ckpt:bool = True
    ckpt_format:str = "_{epoch:02d}-{auroc:.2f}" # debug: try to add formated name
    ckpt_para = { #-----------> paras for pytorch_lightning.callbacks.ModelCheckpoint
                    "save_top_k":1,
                    "mode":"max",
                   "monitor":"auroc",}

    additional_pl_paras={
                #---------> paras for pytorch lightning trainner
                #"accumulate_grad_batches":8, # mil need accumulated grad
                "accelerator":"auto",#accelerator='gpu', devices=1,
            }
    
