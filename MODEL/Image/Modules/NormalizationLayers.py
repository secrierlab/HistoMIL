"""
Simply replace BN2d with Identity2d 

from 
https://github.com/axanderssonuu/ABMIL-ACC
"""
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase


#########################################################################
#   Identity2d
#########################################################################

class _BatchNormIdentity(_NormBase):

    def __init__(self, 
                num_features, 
                eps=1e-5, 
                momentum=0.1, 
                affine=True,
                track_running_stats=True):
        super(_BatchNormIdentity, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return input



class BatchNorm2dIdentity(_BatchNormIdentity):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))