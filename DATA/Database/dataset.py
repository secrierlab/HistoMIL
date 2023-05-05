"""
A simple dataset class for a bag of patches
"""

##############################################################################
#             related import
###############################################################################
import random
import sys
from HistoMIL import logger
from turtle import pos



sys.path.append("..") 
sys.path.append(".") 
from collections import Counter
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
##############################################################################
#             import from other files
###############################################################################
from HistoMIL.DATA.Slide.collector.data_collector import WSICollector,read_wsi_collector
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.FileIO.h5_worker import patchImageStorage
from HistoMIL.DATA.FileIO.pkl_worker import load_pkl
from HistoMIL.EXP.paras.slides import DEFAULT_CONCEPT_PARAS, CollectorParas
from HistoMIL.EXP.paras.dataset import DatasetParas
##############################################################################
#             define MIL Bags for MIL
###############################################################################
class MILFeatureSet(Dataset):
    def __init__(self, data_locs:Locations,
                        data_list:list,
                        label_dict:dict,

                        collector_paras:CollectorParas,
                        sample_nb:int=None):
        self.locs = data_locs
        self.data = data_list

        self.collector_paras = collector_paras
        self.label_dict = label_dict
        self.sample_nb = sample_nb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        [ folder,filename,label] = self.data[idx]
        
        collector = read_wsi_collector(self.locs,
                                    folder, 
                                    filename,
                                    concepts=["feature"],
                                    paras=self.collector_paras,
                                    )
        data = self.processing(collector=collector)
        l = self.label_dict[label]
        del collector
        return data, l
    
    def processing(self,collector:WSICollector):

        # fully select or sample with a number self.sample_nb
        if self.sample_nb is not None:
            idxs = self.get_idxs(full_data=collector.feature.feature_embedding)
            data = torch.index_select(collector.feature.feature_embedding,
                                    dim=0,
                                    index = idxs,
                                    )
        else:
            data = collector.feature.feature_embedding
        return data

    def get_balanced_weight(self,device):
        L = [l for [_,_,l] in self.data]
        L_dict,ratio=get_list_item_count(L,self.label_dict)
        ratio = torch.FloatTensor(ratio).to(device)
        return ratio,L_dict
    
    def get_idxs(self,full_data):
        total_nb = full_data.shape[0]
        full_idx = list(range(total_nb))
        idxs=random.sample(full_idx,self.sample_nb)
        return idxs


# for some use case, we only want read the image once and 
class PatchFeatureset(Dataset):
    def __init__(self, 
                    data_locs:Locations,
                    patch_list:list,
                    label_dict:dict,

                    collector_paras:CollectorParas,
                    ):
        super().__init__()
        self.locs = data_locs
        self.data = patch_list
        
        self.label_dict = label_dict

        self.collector_paras = collector_paras

    def __len__(self):
        return len(self.data)

    def get_balanced_weight(self,device):
        L = [l for [_,_,_,l] in self.data]
        L_dict,ratio=get_list_item_count(L,self.label_dict)
        ratio = torch.FloatTensor(ratio).to(device)
        return ratio,L_dict

    def __getitem__(self, idx):
        
        [ folder,fname,idx,label] = self.data[idx]
        collector = read_wsi_collector(self.locs,
                            folder, 
                            fname,
                            concepts=["feature"],
                            paras=self.collector_paras,)
        coord,data = collector.get_patch_data(name="feature",idx=idx)
        l = self.label_dict[label]
        return data, l

class PatchImageset(Dataset):
    def __init__(self, 
                    data_locs:Locations,
                    patch_list:list,
                    label_dict:dict,
                    concept_paras:CollectorParas=None,
                    # 
                    example_file:str="example/example.svs",
                    img_size=None,
                    trans=None,
                    is_train=True,
                    as_PIL=False,
                    ):
        super().__init__()
        
        self.locs = data_locs
        self.data = patch_list
        self.label_dict = label_dict

        self.concept_paras = concept_paras
        self.set_read_fn(file_name = example_file) # callcable function to get patch  
        self.img_size = img_size
        self.trans = trans(is_train=is_train) if trans is not None else None
        self.is_train = is_train
        self.as_PIL = as_PIL

    def __len__(self):
        return len(self.data)

    def set_read_fn(self,file_name:str):
        file_type = Path(file_name).suffix
        if file_type == ".svs":
            assert self.concept_paras is not None
            self.get_patch_fn = read_one_patch_in_slide
            self.file_para = self.concept_paras
        elif file_type == ".pkl":
            self.get_patch_fn = read_one_patch_in_pkl     
            self.file_para = str(Path(file_name).parent).split("/")[-1]
        if file_type == ".png":
            self.get_patch_fn = read_one_patch_in_img     
            self.file_para = self.concept_paras
        else:
            raise ValueError("file_type not supported")
        #assert self.get_patch_fn is callable

    def get_balanced_weight(self,device):
        L = [l for [_,_,_,l] in self.data]
        L_dict,ratio=get_list_item_count(L,self.label_dict)
        ratio = torch.FloatTensor(ratio).to(device)
        return ratio,L_dict

    def _processing(self,data):
        if self.img_size is not None:
            img = Image.fromarray(data)
            data = img.resize(self.img_size)
            if not self.as_PIL:
                data = np.asarray(data)
        # get transform
        if self.trans is not None:
            data = self.trans(data)#,is_train=self.is_train)#.unsqueeze(0)
        return data

    def __getitem__(self, idx):
        [ folder,fname,idx,label] = self.data[idx]
        coord,data = self.get_patch_fn(self.locs,folder,fname,idx,file_para=self.file_para)
        data = self._processing(data=data)
        l = self.label_dict[label]
        return data, l

def read_one_patch_in_slide(locs,folder,fname,idx,file_para:CollectorParas=DEFAULT_CONCEPT_PARAS):
    collector = read_wsi_collector(locs,
                        folder, 
                        fname,
                        concepts=["slide","patch"],
                        paras=file_para)
    coord,data = collector.get_patch_data(name="patch",idx=idx)
    return coord,data

def read_one_patch_in_pkl(locs,folder,fname,idx,file_para="Patch_Image_512x512"):

    root = locs.abs_loc("img_pkl_loc")+f"/{file_para}/"
    loc = f"{root}/{folder}__{fname}__{str(idx)}.pkl"
    [coord,data] = load_pkl(loc)

    return coord,data

def read_one_patch_in_img(locs,folder,fname,idx,file_para:CollectorParas=DEFAULT_CONCEPT_PARAS):
    #patch_file_name = f"{save_loc}/512_512/{f}_{n}_{j}.png"
    para_folder = f"{file_para.patch.patch_size[0]}_{file_para.patch.step_size}"
    root = locs.abs_loc("patch_img")+f"/{para_folder}/"
    f_name_str = str(Path(fname).stem)
    loc = f"{root}/{folder}_{f_name_str}/{str(idx)}.png"
    # PIL read image into numpy array
    data = np.asarray(Image.open(loc))

    # read coord from collector
    collector = read_wsi_collector(locs,
                        folder, 
                        fname,
                        concepts=["patch"],
                        paras=file_para)
    coord = collector.patch.coords_dict[idx]
    return coord,data

def create_slide_dataset(data_locs:Locations,
                    data_list:list,
                    concept_paras:CollectorParas,
                    dataset_paras:DatasetParas,
                    is_train=True,
                    as_PIL=False,
                    ):
    """
    a single function to create dataset class for different use case
    in:
        data_locs: Locations object
        data_list: list of [folder,fname,idx,label] or [folder,fname,label]
        concepts: list of concepts to read ["slide","patch"] or ["patch"] or ["feature"]
        model_name: model name to read
        label_dict: label dictionary {"label1":0,"label2":1,...}
        is_shuffle: whether to shuffle the data list
    """
    flag = "Train" if is_train else "Val"
    if dataset_paras.is_shuffle:
        random.shuffle(data_list)
    # for different dataset
    if len(data_list[0]) == 4:
        # for patch dataset
        if "feature" in dataset_paras.concepts:
            logger.info(f"Dataset::{flag} Using pre-calculated feature by model:{concept_paras.feature_para.model_name}")
            return PatchFeatureset(data_locs=data_locs,
                                        patch_list=data_list,
                                        label_dict = dataset_paras.label_dict,
                                        collector_paras=concept_paras,
                                        )
        elif "patch" in dataset_paras.concepts and "slide" in dataset_paras.concepts:
            logger.info(f"Dataset::{flag} Using patch image and read from orignal slide.")
            return PatchImageset(data_locs=data_locs,
                                        patch_list=data_list,
                                        label_dict = dataset_paras.label_dict,
                                        concept_paras=concept_paras,
                                        example_file=dataset_paras.example_file,
                                        img_size=dataset_paras.img_size,
                                        trans=dataset_paras.transfer_fn,
                                        is_train=is_train,
                                        as_PIL=as_PIL,
                                        )
        else:
            raise ValueError("concepts should include 'feature' or 'patch'")
    elif len(data_list[0]) == 3:
        # for mil dataset with processed features
        logger.info(f"Dataset::Using pre-calculated feature for MIL.")
        return MILFeatureSet(data_locs=data_locs,
                        data_list=data_list,
                        label_dict = dataset_paras.label_dict,
                        collector_paras=concept_paras,
                        )
    else:
        raise ValueError("Dataset::data_list should be [folder,fname,idx,label] or [folder,fname,label]")


def get_list_item_count(L:list,label_dict:dict):
    length = len(L)
    dict_L = {}
    for key in L:
        dict_L[key] = dict_L.get(key,0)+1 #nb
    for key in dict_L.keys():
        dict_L[key] = dict_L[key]/length #ratio

    if type(label_dict[key])==int:
        weight = []
        list_of_key = list(label_dict.keys())
        list_of_value = list(label_dict.values())
        for i in range(len(dict_L.keys())):
            position = list_of_value.index(i) 
            key_name = list_of_key[position]
            weight.append(1/dict_L[key_name]) #weight
    else:
        logger.warning("Dataset::WARNING! label_dict[key] is not a int value")
        weight = [1/dict_L[key] for key in dict_L.keys() ]
    return dict_L,weight
