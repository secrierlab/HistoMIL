"""
base model for SSL 

include network architecture and step functions for different algorithms
"""
from functools import partial
import copy
import math
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

from HistoMIL import logger
from HistoMIL.MODEL.Image.MIL.utils import FeatureNet
from .paras import SSLParas
from .modules import MLP,BatchRenorm1d
from .utils import BatchShuffleDDP,concat_all_gather



class SSL_base(nn.Module):
    def __init__(self,paras:SSLParas):
        super(SSL_base, self).__init__()
        logger.info(f"init SSL model{paras.ssl_name} with encoder:{paras.encoder_arch}")
        self.model_paras = paras
        self.get_models()

        if self.model_paras.use_negative_examples_from_queue:
            # create the queue
            self.register_buffer("queue", torch.randn(self.model_paras.dim, self.model_paras.K))
            self.queue = torch.nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue = None

    ####################################################################
    #                   get models
    ####################################################################
    def get_models(self):
        self.feat_net = FeatureNet(model_name=self.model_paras.encoder_arch,
                                    pretrained=False)
        self.projection_model = MLP(
            self.model_paras.embedding_dim,
            self.model_paras.dim,
            self.model_paras.mlp_hidden_dim,
            num_layers=self.model_paras.projection_mlp_layers,
            normalization=self.get_norm(),
            weight_standardization=self.model_paras.use_mlp_weight_standardization,
        )
        self.prediction_model = MLP(
                self.model_paras.dim,
                self.model_paras.dim,
                self.model_paras.mlp_hidden_dim,
                num_layers=self.model_paras.prediction_mlp_layers,
                normalization=self.get_norm(prediction=True),
                weight_standardization=self.model_paras.use_mlp_weight_standardization,
            )
        # this classifier is used to compute representation quality each epoch
        self.sklearn_classifier = LogisticRegression(max_iter=100, solver="liblinear")

        if self.model_paras.use_lagging_model:
            # "key" function (no grad)
            self.lagging_model = copy.deepcopy(self.feat_net)
            for param in self.lagging_model.parameters():
                param.requires_grad = False
            # for projection model
            #  "key" function (no grad)
            self.lagging_projection_model = copy.deepcopy(self.projection_model)
            for param in self.lagging_projection_model.parameters():
                param.requires_grad = False
        else:
            self.lagging_model = None
            self.lagging_projection_model = None


    ####################################################################
    #                   step functions for different algorithms
    ####################################################################
    def _get_embeddings(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        bsz, nd, nc, nh, nw = x.shape
        assert nd == 2, "second dimension should be the split image -- dims should be N2CHW"
        im_q = x[:, 0].contiguous()
        im_k = x[:, 1].contiguous()

        # compute query features
        emb_q = self.feat_net(im_q)
        q_projection = self.projection_model(emb_q)
        q = self.prediction_model(q_projection)  # queries: NxC
        if self.model_paras.use_lagging_model:
            # compute key features
            with torch.no_grad():  # no gradient to keys
                if self.model_paras.shuffle_batch_norm:
                    im_k, idx_unshuffle = BatchShuffleDDP.shuffle(im_k)
                k = self.lagging_projection_model(self.lagging_model(im_k))  # keys: NxC
                if self.model_paras.shuffle_batch_norm:
                    k = BatchShuffleDDP.unshuffle(k, idx_unshuffle)
        else:
            emb_k = self.feat_net(im_k)
            k_projection = self.projection_model(emb_k)
            k = self.prediction_model(k_projection)  # queries: NxC

        if self.model_paras.use_unit_sphere_projection:
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)

        return emb_q, q, k

    def _get_contrastive_predictions(self, q, k):
        if self.model_paras.use_negative_examples_from_batch:
            logits = torch.mm(q, k.T)
            labels = torch.arange(0, q.shape[0], dtype=torch.long).to(logits.device)
            return logits, labels

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        if self.model_paras.use_negative_examples_from_queue:
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        return logits, labels

    def _get_pos_neg_ip(self, emb_q, k):
        with torch.no_grad():
            z = self.projection_model(emb_q)
            z = torch.nn.functional.normalize(z, dim=1)
            ip = torch.mm(z, k.T)
            eye = torch.eye(z.shape[0]).to(z.device)
            pos_ip = (ip * eye).sum() / z.shape[0]
            neg_ip = (ip * (1 - eye)).sum() / (z.shape[0] * (z.shape[0] - 1))

        return pos_ip, neg_ip

    ####################################################################
    #                    case queue related functions
    ####################################################################

    def _get_m(self,current_epoch:int=None):
        if self.model_paras.use_momentum_schedule is False:
            return self.model_paras.m
        return 1 - (1 - self.model_paras.m) * (math.cos(math.pi * current_epoch / self.model_paras.max_epochs) + 1) / 2
    
    def _get_temp(self):
        return self.model_paras.T

    @torch.no_grad()
    def _momentum_update_key_encoder(self,current_epoch:int=None):
        """
        Momentum update of the key encoder
        """
        if not self.model_paras.use_lagging_model:
            return
        m = self._get_m(current_epoch=current_epoch)
        for param_q, param_k in zip(self.feat_net.parameters(), self.lagging_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(self.projection_model.parameters(), self.lagging_projection_model.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.model_paras.gather_keys_for_queue:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.model_paras.K % batch_size == 0, f"{self.model_paras.K} % {batch_size} should =0"# for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.model_paras.K  # move pointer

        self.queue_ptr[0] = ptr


    ####################################################################
    #                    architecture functions
    ####################################################################
    def get_norm(self, prediction:bool=False):
        normalization_str = self.model_paras.mlp_normalization
        if prediction and self.model_paras.prediction_mlp_normalization != "same":
            normalization_str = self.model_paras.prediction_mlp_normalization

        if normalization_str is None:
            return None
        elif normalization_str == "bn":
            return partial(torch.nn.BatchNorm1d, num_features=self.model_paras.mlp_hidden_dim)
        elif normalization_str == "br":
            return partial(BatchRenorm1d, num_features=self.model_paras.mlp_hidden_dim)
        elif normalization_str == "ln":
            return partial(torch.nn.LayerNorm, normalized_shape=[self.model_paras.mlp_hidden_dim])
        elif normalization_str == "gn":
            return partial(torch.nn.GroupNorm, num_channels=self.model_paras.mlp_hidden_dim, num_groups=32)
        else:
            raise NotImplementedError(f"mlp normalization {normalization_str} not implemented")
    

####################################################################
#                     
####################################################################
def log_softmax_with_factors(logits: torch.Tensor, log_factor: float = 1, neg_factor: float = 1) -> torch.Tensor:
    exp_sum_neg_logits = torch.exp(logits).sum(dim=-1, keepdim=True) - torch.exp(logits)
    softmax_result = logits - log_factor * torch.log(torch.exp(logits) + neg_factor * exp_sum_neg_logits)
    return softmax_result


def calc_acc_manually(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res