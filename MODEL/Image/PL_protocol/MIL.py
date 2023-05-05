"""
pytorch-lightning wrapper for the model
"""

#---->
import random
import pytorch_lightning as pl

#---->
from HistoMIL import logger
from HistoMIL.MODEL.OptLoss.init import OptLossFactory
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.MODEL.Metrics.init import MetricsFactory
from HistoMIL.EXP.paras.trainer import PLTrainerParas

from HistoMIL.MODEL.Image.PL_protocol.basic import pl_basic
#---->
class pl_MIL(pl_basic):
    def __init__(self,
                data_paras:DatasetParas,# dataset para
                opt_paras:OptLossParas,# optimizer para
                trainer_paras:PLTrainerParas,# trainer para
                model_paras):
        super(pl_MIL,self).__init__()
        """
        A protocol class for abmil follows default setting:
        """
        logger.debug(f"MIL common pl protocol init...")
        #----> paras
        self.data_paras = data_paras
        self.opt_paras = opt_paras
        self.trainer_paras = trainer_paras

        #----> create model
        self.model = None
        self.create_model(model_paras)

        #----> create loss
        self.create_loss()

        #----> create metrics
        self.counts_init(n_classes = self.data_paras.class_nb)
        self.create_metrics()

        logger.debug(f"MIL common pl protocol init done.")

    def create_model(self,model_paras):
        """
        create model instance
        """
        logger.debug(f"PL protocol to create MIL model: {self.trainer_paras.model_name}")
        from HistoMIL.MODEL.Image.MIL.init import aviliable_mil_models
        init_func = aviliable_mil_models[self.trainer_paras.model_name]
        logger.debug(init_func)
        self.model = init_func(model_paras)
    
    def create_loss(self,):
        logger.debug(f"PL protocol to create loss and optimizer with: {self.opt_paras}")
        self.opt_loss_factory = OptLossFactory(para=self.opt_paras,
                                            data_paras=self.data_paras)
        self.opt_loss_factory.create_loss()
        self.loss = self.opt_loss_factory.loss
        
    def create_metrics(self,):
        self.metrics_factory = MetricsFactory(n_classes =self.n_classes,
                                       metrics_names=self.trainer_paras.metric_names)
        self.metrics_factory.get_metrics(task_type=self.trainer_paras.task_type)

        self.bar_metrics = self.metrics_factory.metrics["metrics_on_bar"]
        self.valid_metrics = self.metrics_factory.metrics["metrics_template"].clone(prefix = 'val_')
        self.test_metrics = self.metrics_factory.metrics["metrics_template"].clone(prefix = 'test_')

    def configure_optimizers(self):
        optimizer =self.opt_loss_factory.create_optimizer(self.model.parameters())
        scheduler = self.opt_loss_factory.create_scheduler(optimizer)
        return [optimizer],[scheduler]

    def training_step(self, batch, batch_idx):
        #---->model step
        data, label = batch
        #print(data.shape)
        results_dict = self.model(data)

        #------> check output is valid
        self.confirm_model_outputs(results_dict, self.trainer_paras.model_out_list)
        #---->confirm label format
        label_Y = label
        #print(results_dict,label_Y)
        #---->loss step
        loss = self.loss(results_dict['logits'], label_Y)

        #---->overall counts log 
        self.counts_step(Y_hat=results_dict['Y_hat'], label=label_Y, train_phase="train")
        return {'loss': loss} 

    def training_epoch_end(self, training_step_outputs):
        self.counts_end_and_log(train_phase="train")

    def validation_step(self, batch, batch_idx):
        #---->model step
        data, label = batch
        results_dict = self.model(data)
        
        #---->confirm label format
        label_Y = label
        results_dict.update({'label':label})

        #print(results_dict['Y_hat'],label_Y)
        #---->overall counts log 
        self.counts_step(Y_hat=results_dict['Y_hat'], label=label_Y, train_phase="valid")
        return results_dict
    
    def validation_epoch_end(self, validation_step_outputs):
        self.log_val_metrics(validation_step_outputs,
                            bar_name = self.metrics_factory.metrics_names[0])
        #---->counts log
        self.counts_end_and_log(train_phase="valid")
        #----> shuffle data
        self.re_shuffle(self.trainer_paras.shuffle_data)
