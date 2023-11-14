"""
Implementation of abmil with/without gradient accumulation

most code copy from 
https://github.com/szc19990412/TransMIL
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#-----------> external network modules 
from HistoMIL.MODEL.Image.Modules.NystromAttention import NystromAttention
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet
from HistoMIL.MODEL.Image.MIL.TransMIL.paras import TransMILParas
from HistoMIL import logger
######################################################################################
#        pytorch module to define structure
######################################################################################
class TransMIL(nn.Module):
    def __init__(self,paras:TransMILParas):
        super(TransMIL, self).__init__()
        logger.info(f"init TransMIL with paras: {paras}")
        self.paras = paras
        backbone = paras.encoder_name
        pretrained = paras.encoder_pretrained
        feature_size = paras.feature_size
        embed_size = paras.embed_size if paras.embed_size is not None else int(feature_size//2)
        n_classes = paras.class_nb
        norm_layer = paras.norm_layer
        #--------> feature encoder
        self.encoder = FeatureNet(backbone,pretrained=pretrained)
        #--------> transformer aggregation
        self.pos_layer = PPEG(dim=embed_size)
        self._fc1 = nn.Sequential(nn.Linear(feature_size, embed_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=embed_size,norm_layer=norm_layer)
        self.layer2 = TransLayer(dim=embed_size,norm_layer=norm_layer)
        self.norm = nn.LayerNorm(embed_size)
        self._fc2 = nn.Linear(embed_size, self.n_classes)


    def forward(self, data):
        #--------> feature encoder
        data = self.encoder(data)

        #--------> transformer aggregation
        h = data.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        #print(h.shape)
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        #print(h.shape)
        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]
        #print(h.shape)
        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        #print(h.shape)
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        #print(h.shape)
        #---->cls_token
        f = self.norm(h)[:,0]
        #print(h.shape)
        #---->predict
        logits = self._fc2(f) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,"att":h}
        return results_dict

######################################################################################
#        modules for trans_mil
######################################################################################

class TransLayer(nn.Module):

    def __init__(self,dim , norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.5
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x