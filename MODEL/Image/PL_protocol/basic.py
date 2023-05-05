"""
basic design in PL protocol
"""
import torch
import random
import pytorch_lightning as pl
from HistoMIL.MODEL.Image.PL_protocol.utils import current_label_format,label_format_transfer

class pl_basic(pl.LightningModule):
    def __init__(self):
        super(pl_basic,self).__init__()
        """
        A basic class for different PL protocols:
        """
        pass
    def counts_init(self,n_classes):
        self.n_classes = n_classes
        self.tr_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.val_data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.count = 0
    
    def counts_step(self,Y_hat,label,train_phase:str="train"):
        data = self.tr_data if train_phase=="train" else self.val_data
        for i in range(label.shape[0]):
            #for j in range(self.n_classes):
            Y = int(label[i].item())
            data[Y]["count"] += 1
            num_correct = torch.eq(Y_hat[i,...], label[i]).sum().float().item()
            data[Y]["correct"] += (num_correct)
        if train_phase=="train": self.tr_data = data
        else: self.val_data = data

    def counts_end_and_log(self,train_phase:str="train"):
        data = self.tr_data if train_phase=="train" else self.val_data
        all_acc = 0.0
        for c in range(self.n_classes):
            count = data[c]["count"]
            correct = data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
                all_acc+=acc
            print('{}: class {}: acc {}, correct {}/{}'.\
                        format(train_phase, c, acc, correct, count))
        data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        if train_phase=="train": 
            self.tr_data = data
            self.log("Train ACC",all_acc/self.n_classes)
        else: 
            self.val_data = data
    def re_shuffle(self,is_shuffle:bool=True):
        #---->random, if shuffle data, change seed
        if is_shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def confirm_label_format(self,
                                label,
                                target_format,
                                target_task:str="classification",
                                ):
        #no longer need to check label format with trainer setting
        return label
    
    def confirm_model_outputs(self,
                                model_outputs:dict,
                                target_outputs:list=["logits","Y_prob","Y_hat"],
                                ):
        #----> check model output
        for key in target_outputs:
            if key not in model_outputs.keys():
                raise ValueError(f"model output {key} not in model_outputs")

    def collect_step_output(self,key,out,dim=None):
        if dim is None:
            return torch.cat([x[key] for x in out])
        else:
            return torch.cat([x[key] for x in out],dim=dim)

    def log_val_metrics(self,outlist,bar_name:str):
        probs = self.collect_step_output(key="Y_prob",out=outlist,dim=0)
        max_probs = self.collect_step_output(key="Y_hat",out=outlist,dim=0)
        target = self.collect_step_output(key="label",out=outlist,dim=0)
        #----> log part
        self.log(bar_name, self.bar_metrics(probs, target.squeeze()), 
                            prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)
