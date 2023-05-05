"""
Implementation of abmil with/without gradient accumulation

most code copy from 
https://github.com/axanderssonuu/ABMIL-ACC
"""
import timm 
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck

import random

from HistoMIL import logger
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet
from HistoMIL.MODEL.Image.MIL.ABMIL.paras import AttMILParas
############################################################
#         basic modules
############################################################

class ABMIL(torch.nn.Module):
    '''
        The MILModel consists of two parts:
            1. A feature extractor model that, given 
                a patch, or batch of patches, outputs a 
                feature vector.
            2. An attention model that that outputs a
                prediction given a set of feature vectors.
    '''

    def __init__(self, paras:AttMILParas):
        super(ABMIL,self).__init__()

        self.paras = paras
        logger.info(f"ABMIL model will be built with {paras}")
        backbone = paras.encoder_name
        pretrained = paras.encoder_pretrained
        feature_dim = paras.feature_dim
        hidden_dim = paras.hidden_dim
        n_classes = paras.class_nb
        logger.info(f"The encoder network will be set as {backbone}")
        self.feature_extractor = FeatureNet(model_name=backbone,
                                            pretrained=pretrained)
        logger.debug(f"The AttentionNet will be set as {feature_dim}: {hidden_dim} :{n_classes}")
        self.attention_model = AttentionNet(feature_dim=feature_dim,
                                            hidden_dim=hidden_dim,
                                            class_nb=n_classes)
        self.last_activation = nn.Sigmoid()

        self.train_pattern = "attention" # or "feature" or "all"
        logger.debug(f"ABMIL model init finished.")
        
    def set_trainable(self,):
        assert self.train_pattern in ["attention","feature","all","none"]
        self.feature_extractor.freeze()
        self.attention_model.freeze()
        if self.train_pattern=="attention":
            self.attention_model.unfreeze()
        elif self.train_pattern=="feature":
            self.feature_extractor.unfreeze()
        elif self.train_pattern=="all":
            self.feature_extractor.unfreeze()
            self.attention_model.unfreeze()

    def sample_instance(self,x):
        '''
        sample a batch of features from calculated bag
            x: a batch of patches with shape (batch_size, feature_dim)
        '''
        if (x.shape[1] <= self.paras.sample_nb) or (self.paras.sample_nb <=0):
            return x
        else:
            indice = random.sample(range(x.shape[1]), self.paras.sample_nb)
            indice = torch.tensor(indice)
            sampled_values = x[:,indice,:]
            return sampled_values

    def forward(self, x):
        '''
            x: a batch of patches
        '''
        self.set_trainable()

        x = self.feature_extractor(x)
        x = self.sample_instance(x)
        logits, attention = self.attention_model(x)
        if self.last_activation is not None:
            logits = self.last_activation(logits)
        #---->predict
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 
                        'Y_hat': Y_hat,"att":attention}
        return results_dict
    
    def update_feature_extractor(self,batch_of_patches,patch_idx,
                                    no_grad_fv,identity_matrix):
        fv = self.feature_extractor(batch_of_patches) # with gradient calculate features
        other_patches_idx = ~torch.max(identity_matrix[patch_idx], dim=0)[0]
        other_fv = no_grad_fv[other_patches_idx]
        pred, _ = self.attention_model(torch.cat( (fv, other_fv)))
        return pred


############################################################
#         basic modules
############################################################

class AttentionNet(torch.nn.Module):
    def __init__(self,feature_dim:int = 2048,hidden_dim:int=512,class_nb:int=1):
        super(AttentionNet, self).__init__()
        self.L = feature_dim#2048 
        self.D = hidden_dim#524  
        self.K = class_nb #1
        #print(f"AttentionNet will be built with L={self.L}, D={self.D}, K={self.K}")
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.K),
            #nn.Sigmoid()
        )

        self.freeze_flag = False
        
    def freeze(self):
        if not self.freeze_flag:
            for params in self.parameters():
                params.requires_grad = False
            self.freeze_flag = True
        
    def unfreeze(self):
        if self.freeze_flag:
            for param in self.parameters():
                param.requires_grad = True
            self.freeze_flag = False

    def forward(self, H):
        #remove batch dim,normally batch size is 1 
        H = H.squeeze()
        # will not affect the result
        # calculate attention
        A = self.attention(H)
        
        A = torch.transpose(A, 0, 1)
        
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, H) # [self.L,dim] x [dim,self.K] = [self.L,self.K]
        M = M.flatten().unsqueeze(0) # flatten to 1D [1,self.L*self.K]
        
        Y_prob = self.classifier(M) #[B, n_classes]

        return Y_prob, A
    


