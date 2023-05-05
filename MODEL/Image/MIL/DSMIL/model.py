"""
Implement of model dsmil
https://github.com/binli123/dsmil-wsi/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchsnooper

from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet
########################################################################################
# .     modules in DSMIL
########################################################################################
class PClassifier(nn.Module):
    def __init__(self, feature_size, output_class,dropout_r=0.0):
        super(PClassifier, self).__init__()
    
        #self.fc = nn.Linear(feature_size, output_class)

        #self.fc = nn.Sequential(
        #                nn.Linear(feature_size, output_class),
        #                nn.Dropout(dropout_r),
        #                nn.Softmax()
        #                )
        self.fc = nn.Sequential(
                nn.Linear(feature_size, int(feature_size//2)),
                nn.Dropout(dropout_r),
                nn.ReLU(inplace=True),
                nn.Linear(int(feature_size//2), output_class),
                )

    #@torchsnooper.snoop()
    def forward(self, x):
        #x = x.view(x.shape[0], -1) # N x C
        c = self.fc(x) # N x Class
        return c

# https://github.com/binli123/dsmil-wsi/blob/ee7010be9dcc4b608ea980c9f10da59d845a4b0c/dsmil.py
class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_r=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_r),
            nn.Linear(input_size, input_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
    
    #@torchsnooper.snoop()
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        # change forward function remove this part
        #C = F.softmax(C,dim=1)
        return C, A, B 

########################################################################################
# .     model definition for DSMIL
########################################################################################

class DSMIL(torch.nn.Module):
    def __init__(self,paras:DSMILParas):
        super(DSMIL, self).__init__()
        self.paras = paras
        self.feature_extractor = FeatureNet(model_name=paras.encoder_name,
                                            pretrained=paras.encoder_pretrained)

        self.patch_classifier = PClassifier(feature_size=paras.feature_dim, 
                                            output_class= paras.p_class,
                                            dropout_r=paras.dropout_r)
        self.bag_classifier = BClassifier(input_size=paras.feature_dim, 
                                            output_class=paras.b_class,
                                            dropout_r=paras.dropout_r)


    #@torchsnooper.snoop()
    def step_patch(self,x):
        p_c  = self.patch_classifier(x)
        return p_c, x.view(x.shape[0],-1)


    #@torchsnooper.snoop()
    def forward(self, x):
        #---->extract feature
        x = self.feature_extractor(x)
        #---->predict patch
        #x : Batch x Nb of instance x Feature
        x = x.squeeze(0) # [1xN,F] Batch=1 in MIL
        p_c  = self.patch_classifier(x)

        #---->predict bag
        b_logits,A,B = self.bag_classifier(x,p_c) #[B, n_classes]

        #---->predict
        #logits=self.classifier(feat) #[B, n_classes]
        Y_hat = torch.argmax(b_logits, dim=1)
        Y_prob = F.softmax(b_logits, dim = 1)
        results_dict = {'logits': b_logits,'Y_prob': Y_prob, 'Y_hat': Y_hat,"patch_pred":p_c,"attention":A,"feature":B}
        return results_dict

