"""
pytorch-lightning wrapper for the model
"""

#---->
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
#---->
from HistoMIL import logger
from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.MODEL.Image.MIL.ABMIL.paras import AttMILParas
#---->
####################################################################################
#      pl protocol class
####################################################################################
class  pl_ABMIL(pl_MIL):
    #---->init
    def __init__(self, 
                data_paras:DatasetParas,# dataset para
                opt_paras:OptLossParas,# optimizer para
                trainer_paras:PLTrainerParas,# trainer para
                model_para:AttMILParas):
        super(pl_ABMIL, self).__init__(data_paras,
                                    opt_paras,
                                    trainer_paras,
                                    model_para)
        """
        model:: model instance of abmil
        loss:: name of different loss function
        optimizer:: 
        """
        self.scale_att = self.trainer_paras.model_para.loss_scale_att
        self.scale_feat = self.trainer_paras.model_para.loss_scale_feat

        self.train_feat = self.trainer_paras.model_para.update_feature_extractor
        self.max_instances = self.trainer_paras.model_para.max_instances
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        logger.debug(f"ABMIL pl protocol init done.")
        pass

    def split_into_batches(self,iterable, n=1):
        l = len(iterable)
        bag = [iterable[ndx:min(ndx + n, l)] for ndx in range(0, l, n)]
        idx = [ torch.arange(ndx,min(ndx+n,len(bag))) for ndx in range(0, len(bag), n)]
        return bag, idx

    def get_no_grad_fv(self,bag):
        """
        get the feature vector of the model without gradient
        """
        #---->get the feature vector of the model without gradient
        self.model.train_pattern="none" # set the model to eval mode
        self.model.set_trainable()
        no_grad_fv,identity = self.model.feature_extractor.get_features_bag(bag)
        return no_grad_fv,identity

    def update_att_net(self,fv,target):
        self.model.train_pattern="att" # set the model to train mode
        self.model.set_trainable()
        pred, attention = self.model.attention_model(fv)
        l = self.loss(pred, target) * self.scale_att
        l.backward()
        return pred,attention

    def update_feature_net(self,bag,target,no_grad_fv,identity):
        bag, idx =self.split_into_batches(bag, n=self.max_instances)
        # set feature extractor to train mode and not update the attention network
        self.model.train_pattern="feature" # set the model to train mode
        self.model.set_trainable()

        for patches,patch_idx in (zip(bag,idx)):
            pred=self.model.update_feature_extractor(self,patches,patch_idx,
                                    no_grad_fv,identity)
            l = self.loss(pred, target) * self.scale_feat
            l.backward()

    def training_step(self, batch, batch_idx):
        #---->model step
        data, label = batch
        #---->confirm label format
        label_Y = label
        #print(data.shape)
        opt = self.optimizers()
        # single scheduler
        sch = self.lr_schedulers()

        if self.train_feat:
            opt.zero_grad()
            no_grad_fv,identity=self.get_no_grad_fv(data)
            logits,att = self.update_att_net(no_grad_fv,label_Y)
            self.update_feature_net(data,label_Y,
                                    no_grad_fv,identity)

            #---->predict
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim = 1)
            results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,"att":att}
            
        else:
            results_dict = self.model(data)
            #---->loss step
            loss = self.loss(results_dict['logits'], label_Y)


        #------> check output is valid
        self.confirm_model_outputs(results_dict, self.trainer_paras.model_out_list)

        #---->opt step
        opt.step()
        sch.step()
        opt.zero_grad()
        #---->overall counts log 
        self.counts_step(Y_hat=results_dict['Y_hat'], label=label_Y, train_phase="train")
        return {'loss': loss} 

