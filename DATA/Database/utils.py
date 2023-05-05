# get weight sampler
import torch
from HistoMIL import logger
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


def get_weight_sampler(dataset,label_np=None):
    label_np = dataset[:][1].numpy() if label_np is None else label_np
    class_sample_count = np.array(
            [len(np.where(label_np==t)[0]) for t in np.unique(label_np)])
    weight = np.array([1-(item/sum(class_sample_count)) for item in class_sample_count])
    logger.info(f"Dataset:: Current dataset with class count{class_sample_count} will be sampled as weight {weight}")
    L = label_np.tolist()
    samples_weight = np.array([weight[int(item)] for item in L])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), label_np.shape[0])
    return sampler

