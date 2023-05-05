import torch
import numpy as np
def label_format_transfer(target_format,label_dict):
    if target_format=="one_hot":
        # label_dict from int to one_hot
        new_label_dict = {}
        keys = list(label_dict.keys())
        base = [0 for i in range(len(keys))]
        for i,k in enumerate(keys):
            one_hot = base.copy()
            one_hot[i] = 1
            new_label_dict.update({k:one_hot})
        label_dict = new_label_dict
    elif target_format=="int":
        # label_dict from one_hot to int
        new_label_dict = {}
        keys = list(label_dict.keys())
        for i,k in enumerate(keys):
            new_label_dict.update({k:i})
        label_dict = new_label_dict
    else:
        raise ValueError(f"target_format {target_format} not supported")
    return label_dict

def current_label_format(label,task:str="classification"):
    # for the label not as np format change to np format
    if task == "classification":
        if len(label.shape) == 1 : current_format = "int"
        elif label.shape[1] > 1: current_format = "onehot" #shape [batch x n_classes]   
        else: current_format = "int"
    elif task == "regression": #shape [batch x 1]
        current_format = "float"
    return current_format