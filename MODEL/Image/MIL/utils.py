import torch
import timm
from HistoMIL import logger
#--------> commonly used feature encoder function for MIL

class FeatureNet(torch.nn.Module):
    def __init__(self,model_name,pretrained:bool=True):
        super(FeatureNet, self).__init__()
        logger.info(f"FeatureNet:: Use: {model_name} ")
        self.name = model_name
        if model_name == "pre-calculated":
            self.pre_trained = None
        else:# get pretrained model for feature extraction
            self.pre_trained = timm.create_model(model_name, 
                                            pretrained=pretrained, 
                                            num_classes=0)
        self.freeze_flag = False

    def freeze(self):
        if self.pre_trained is not None:
            if not self.freeze_flag:
                for params in self.parameters():
                    params.requires_grad = False
                
                for layer in self.modules():
                    if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                        layer.track_running_stats = False

                self.freeze_flag = True
        
    def unfreeze(self):
        if self.pre_trained is not None:
            if self.freeze_flag:
                for param in self.parameters():
                    param.requires_grad = True
                for layer in self.modules():
                    if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
                        layer.track_running_stats = True
                self.freeze_flag = False

    #@torchsnooper.snoop()
    def forward(self, x):
        if self.pre_trained is None:
            return x
        x = self.pre_trained(x)
        return x

    def get_features_bag(self, bag):
        with torch.no_grad():
            fv = []
            for batch_of_patches in bag:
                fv.append(self.forward(batch_of_patches))
            identity_matrix = torch.eye(fv.shape[0], dtype=torch.bool)
            return torch.cat(fv),identity_matrix