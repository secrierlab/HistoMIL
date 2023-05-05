
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from functools import partial

#---->
from HistoMIL import logger
from HistoMIL.MODEL.OptLoss.init import OptLossFactory
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas
from HistoMIL.MODEL.Metrics.init import MetricsFactory
from HistoMIL.EXP.paras.trainer import PLTrainerParas


from .paras import SSLParas
from .base_model import SSL_base,log_softmax_with_factors,calc_acc_manually
from .modules import LARS
#---->
class pl_SSL(pl.LightningModule):
    def __init__(self,
                 model_paras:SSLParas,# model para
                ):
        super(pl_SSL,self).__init__()
        #----> model 
        self.model_paras = model_paras
        self.model = SSL_base(model_paras)

        self.optloss_para = self.model_paras.ssl_opt_loss_para
        
        self.dataset_para = self.model_paras.ssl_dataset_para
        #
        some_negative_examples = self.model_paras.use_negative_examples_from_batch or self.model_paras.use_negative_examples_from_queue

    ################################################################
    #
    ################################################################
    def configure_optimizers(self):
        # exclude bias and batch norm from LARS and weight decay
        regular_parameters = []
        regular_parameter_names = []
        excluded_parameters = []
        excluded_parameter_names = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad is False:
                continue
            if any(x in name for x in self.model_paras.exclude_matching_parameters_from_lars):
                excluded_parameters.append(parameter)
                excluded_parameter_names.append(name)
            else:
                regular_parameters.append(parameter)
                regular_parameter_names.append(name)

        param_groups = [
            {"params": regular_parameters, "names": regular_parameter_names, "use_lars": True},
            {
                "params": excluded_parameters,
                "names": excluded_parameter_names,
                "use_lars": False,
                "weight_decay": 0,
            },
        ]

        if self.optloss_para.Opt_name == "sgd":
            optimizer = torch.optim.SGD
        elif self.optloss_para.Opt_name == "lars":
            optimizer = partial(LARS, 
                                warmup_epochs=self.optloss_para.opt_paras["lars_warmup_epochs"],
                                eta=self.optloss_para.opt_paras["lars_eta"])
        else:
            raise NotImplementedError(f"SSL part not support {self.optloss_para.Opt_name}")

        encoding_optimizer = optimizer(
            param_groups,
            lr=self.optloss_para.lr,
            momentum=self.optloss_para.opt_paras["momentum"],
            weight_decay=self.optloss_para.opt_paras["weight_decay"],
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            encoding_optimizer,
            self.optloss_para.max_epochs,
            eta_min=self.optloss_para.scheduler_paras["final_lr_schedule_value"],
        )
        return [encoding_optimizer], [self.lr_scheduler]


    ################################################################
    # special loss function
    ################################################################
    def _get_contrastive_loss(self, logits, labels):
        loss_name = self.optloss_para.Loss_name[0] 
        assert loss_name in ["CrossEntropyLoss", "BCELoss", "InnerProduct"] 
        if loss_name == "CrossEntropyLoss":
            if self.model_paras.use_eqco_margin:
                if self.model_paras.use_negative_examples_from_batch:
                    neg_factor = self.model_paras.eqco_alpha / self.dataset_para.batch_size
                elif self.model_paras.use_negative_examples_from_queue:
                    neg_factor = self.model_paras.eqco_alpha / self.model_paras.K
                else:
                    raise Exception("Must have negative examples for CrossEntropyLoss")

                predictions = log_softmax_with_factors(logits / self.model_paras.T, neg_factor=neg_factor)
                return F.nll_loss(predictions, labels)

            return F.cross_entropy(logits / self.model_paras.T, labels)

        new_labels = torch.zeros_like(logits)
        new_labels.scatter_(1, labels.unsqueeze(1), 1)
        if loss_name == "BCELoss":
            return F.binary_cross_entropy_with_logits(logits / self.model_paras.T, new_labels) * logits.shape[1]

        if loss_name == "InnerProduct":
            # inner product
            # negative sign for label=1 (maximize ip), positive sign for label=0 (minimize ip)
            inner_product = (1 - new_labels * 2) * logits
            return torch.mean((inner_product + 1).sum(dim=-1))

        raise NotImplementedError(f"Loss function {self.optloss_para.Loss_name} not implemented")

    def _get_vicreg_loss(self, z_a, z_b, batch_idx):
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2
        assert self.model_paras.variance_loss_epsilon >=0
        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.model_paras.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.model_paras.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.model_paras.invariance_loss_weight
        weighted_var = loss_var * self.model_paras.variance_loss_weight
        weighted_cov = loss_cov * self.model_paras.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return {
            "loss": loss,
            "loss_invariance": weighted_inv,
            "loss_variance": weighted_var,
            "loss_covariance": weighted_cov,
        }
    
    ################################################################
    #    training step
    ################################################################
    def forward(self, x):
        return self.model.feat_net(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        all_params = list(self.model.feat_net.parameters())
        x, class_labels = batch  # batch is a tuple, we just want the image

        emb_q, q, k = self.model._get_embeddings(x)
        pos_ip, neg_ip = self.model._get_pos_neg_ip(emb_q, k)

        logits, labels = self.model._get_contrastive_predictions(q, k)
        if self.model_paras.use_vicreg_loss:
            losses = self._get_vicreg_loss(q, k, batch_idx)
            contrastive_loss = losses["loss"]
        else:
            losses = {}
            contrastive_loss = self._get_contrastive_loss(logits, labels)

            if self.model_paras.use_both_augmentations_as_queries:
                x_flip = torch.flip(x, dims=[1])
                emb_q2, q2, k2 = self.model._get_embeddings(x_flip)
                logits2, labels2 = self.model._get_contrastive_predictions(q2, k2)

                pos_ip2, neg_ip2 = self.model._get_pos_neg_ip(emb_q2, k2)
                pos_ip = (pos_ip + pos_ip2) / 2
                neg_ip = (neg_ip + neg_ip2) / 2
                contrastive_loss += self._get_contrastive_loss(logits2, labels2)

        contrastive_loss = contrastive_loss.mean() * self.model_paras.loss_constant_factor

        log_data = {
            "step_train_loss": contrastive_loss,
            "step_pos_cos": pos_ip,
            "step_neg_cos": neg_ip,
            **losses,
        }

        with torch.no_grad():
            self.model._momentum_update_key_encoder(self.current_epoch)

        some_negative_examples = (
            self.model_paras.use_negative_examples_from_batch or self.model_paras.use_negative_examples_from_queue
        )
        if some_negative_examples:
            acc1, acc5 = calc_acc_manually(logits, labels, topk=(1, 5))
            log_data.update({"step_train_acc1": acc1, "step_train_acc5": acc5})

        # dequeue and enqueue
        if self.model_paras.use_negative_examples_from_queue:
            self.model._dequeue_and_enqueue(k)

        self.log_dict(log_data)
        return {"loss": contrastive_loss}


    def validation_step(self, batch, batch_idx):
        x, class_labels = batch
        with torch.no_grad():
            emb = self.model.feat_net(x)

        return {"emb": emb, "labels": class_labels}

    def validation_epoch_end(self, outputs):
        embeddings = torch.cat([x["emb"] for x in outputs]).cpu().detach().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().detach().numpy()
        num_split_linear = embeddings.shape[0] // 2
        self.model.sklearn_classifier.fit(embeddings[:num_split_linear], labels[:num_split_linear])
        train_accuracy = self.model.sklearn_classifier.score(embeddings[:num_split_linear], labels[:num_split_linear]) * 100
        valid_accuracy = self.model.sklearn_classifier.score(embeddings[num_split_linear:], labels[num_split_linear:]) * 100

        log_data = {
            "epoch": self.current_epoch,
            "sklearn_tr_acc": train_accuracy,
            "sklearn_val_acc": valid_accuracy,
            "T": self.model._get_temp(), # temperature
            "m": self.model._get_m(),   # momentum
        }
        logger.info(f"Epoch {self.current_epoch} accuracy: sk train: {train_accuracy:.1f}%, sk validation: {valid_accuracy:.1f}%")
        self.log_dict(log_data)


    ################################################################
    #
    ################################################################