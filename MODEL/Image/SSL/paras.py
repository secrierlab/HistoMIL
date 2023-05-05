"""
pre-defined parameters for SSL training
"""
import torch.nn as nn
import attr 
from functools import partial
from typing import List
from typing import Optional

from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.dataset import DatasetParas
#---->
@attr.s(auto_attribs=True)
class SSLParas(object):
    #----->  model parameters
    ssl_name: str = "moco"
    # encoder parameters
    encoder_arch: str = "resnet18"
    shuffle_batch_norm: bool = False
    embedding_dim: int = 512  # must match embedding dim of encoder
    # MLP parameters
    projection_mlp_layers: int = 2
    prediction_mlp_layers: int = 0
    mlp_hidden_dim: int = 512

    mlp_normalization: Optional[str] = None # "bn", "br", "ln", "gn"
    prediction_mlp_normalization: Optional[str] = "same"  # if same will use mlp_normalization
    use_mlp_weight_standardization: bool = False

    #-----> optimizer parameters
    """
    # optimization parameters
    lr: float = 0.5
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_epochs: int = 320
    final_lr_schedule_value: float = 0.0
    """
    ssl_opt_loss_para: OptLossParas = OptLossParas(
                                            Loss_name=["CrossEntropyLoss",],
                                            #loss_type only support ["CrossEntropyLoss","BCELoss","InnerProduct"]
                                            Opt_name="sgd", #only ["sgd","lars"]
                                            lr=0.5,
                                            opt_paras={"momentum":0.9,
                                                        "weight_decay":1e-4,
                                                        },
                                            # for lars
                                            #opt_paras={
                                            #     "momentum":0.9,
                                            #     "weight_decay":1e-4,
                                            #     "lars_warmup_epochs" : 1, #int
                                            #     "lars_eta": 1e-3, #float
                                            # }
                                            max_epochs=320,
                                            scheduler_paras={"final_lr_schedule_value":0.0})

    #-----> data-related parameters
    ssl_dataset_para: DatasetParas = DatasetParas(
                                            concepts = ["slide","patch"],
                                            batch_size=128,
                                            num_workers=64,
                                            is_weight_sampler=False,
                                            )
    # transform parameters
    transform_s: float = 0.5
    transform_apply_blur: bool = True
    # data loader parameters
    drop_last_batch: bool = True
    pin_data_memory: bool = True
    gather_keys_for_queue: bool = False
    #-----> SSL parameters
    # MoCo parameters
    K: int = 65536  # number of examples in queue
    dim: int = 128
    m: float = 0.996
    T: float = 0.2

    # eqco parameters
    eqco_alpha: int = 65536
    use_eqco_margin: bool = False
    use_negative_examples_from_batch: bool = False

    # Change these to make more like BYOL
    use_momentum_schedule: bool = False
    #loss_type: str = "ce"
    use_negative_examples_from_queue: bool = True
    use_both_augmentations_as_queries: bool = False
    #optimizer_name: str = "sgd"

    exclude_matching_parameters_from_lars: List[str] = []  # set to [".bias", ".bn"] to match paper
    loss_constant_factor: float = 1

    # Change these to make more like VICReg
    use_vicreg_loss: bool = False
    use_lagging_model: bool = True
    use_unit_sphere_projection: bool = True
    invariance_loss_weight: float = 25.0
    variance_loss_weight: float = 25.0
    covariance_loss_weight: float = 1.0
    variance_loss_epsilon: float = 1e-04

    



# Differences between these parameters and those used in the paper (on image net):
# max_epochs=1000,
# lr=1.6,
# batch_size=2048,
# weight_decay=1e-6,
# mlp_hidden_dim=8192,
# dim=8192,
VICRegParams = partial(
    SSLParas,
    ssl_name="vicreg",
    use_vicreg_loss=True,
    loss_type="vic",
    use_lagging_model=False,
    use_unit_sphere_projection=False,
    use_negative_examples_from_queue=False,
    optimizer_name="lars",
    exclude_matching_parameters_from_lars=[".bias", ".bn"],
    projection_mlp_layers=3,
    final_lr_schedule_value=0.002,
    mlp_normalization="bn",
    lars_warmup_epochs=10,
)

BYOLParams = partial(
    SSLParas,
    ssl_name="byol",
    prediction_mlp_layers=2,
    mlp_normalization="bn",
    loss_type="ip",
    use_negative_examples_from_queue=False,
    use_both_augmentations_as_queries=True,
    use_momentum_schedule=True,
    optimizer_name="lars",
    exclude_matching_parameters_from_lars=[".bias", ".bn"],
    loss_constant_factor=2,
)

SimCLRParams = partial(
    SSLParas,
    ssl_name="simclr",
    use_negative_examples_from_batch=True,
    use_negative_examples_from_queue=False,
    use_lagging_model=False,
    K=0,
    m=0.0,
    use_both_augmentations_as_queries=True,
)