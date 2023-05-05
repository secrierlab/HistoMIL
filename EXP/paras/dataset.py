"""


"""

import attr

from HistoMIL.DATA.Database.data_aug import naive_transforms

@attr.s(auto_attribs=True)
class DatasetParas(object):
    dataset_name:str = None #normally the same as the cohort_paras.task_name
    #----> meta data for a dataset
    concepts: list = None

    data_len:int=None
    class_nb:int=None
    category_ratio:list=None #list of ratio for different categories
    imbalance_loss_weight:list = None
    sampler_weight:list=None
    label_dict:dict=None    # same as cohort label_dict

    example_file:str="example/example.svs"

    #----> data split for training and testing
    split_ratio:list=None
    #----> for dataloader
    batch_size:int=1 # normally 1 for mil, 8 for patch_learning
    is_shuffle:bool=True
    is_weight_sampler:bool=True
    num_workers:int=0

    force_balance_val:bool=False
    over_sample:float=1.0
    add_dataloader:dict = None#{"pin_memory":bool=True,"drop_last":bool=False}
    #----> for data augmentation
    img_size:tuple=(512,512)
    transfer_fn:callable=naive_transforms #no_transforms
    is_train:bool=None # None means select by training phase, True means train, False means test
    as_PIL:bool=False
    storage_file:str=None#"example/example.svs"
    add_data_aug_paras:dict = {
        "resize" : 512,
    }
    #----> for additional data process
    method_additional_paras:dict=None
