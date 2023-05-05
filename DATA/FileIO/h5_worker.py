"""
A simple h5 file read/write manager

"""
import h5py
from HistoMIL import logger
import numpy as np
from pathlib import Path

class h5Abstract:
    def __init__(self, loc:str)-> None:
        self.loc = loc

    def create(self,map_func,paras=None) -> None:
        with h5py.File(self.loc, mode="w") as h5f:
            map_func(h5f,paras)
    
    def read(self,map_func,paras=None) -> None:
        with h5py.File(self.loc, mode="r") as h5f:
            out = map_func(h5f,paras)
        return out
    
    def add(self,map_func,paras=None) -> None:
        with h5py.File(self.loc, mode="a") as h5f:
            map_func(h5f,paras)

"""
# simple example for map_func design in h5Abstract
def map_func(h5f,paras=None) -> dict:
    # get paras needed
    ...
    # process logic
    ...
    # return result
    ...
"""

class patchCoordsStorage(h5Abstract):
    def __init__(self, loc) -> None:
        super().__init__(loc)

    def read(self,key,is_attrs,attrs=[]):
        def map_func(h5f,paras):
            # get paras needed
            key = paras["key"]
            is_attrs = paras["is_attrs"]
            attrs = paras["attrs"]
            # process logic
            if is_attrs:
                out={}
                assert len(attrs)>0
                for item in attrs:
                    data = h5f[key].attrs[item]
                    out.update({item:data})
                #out = f[key].attrs
            else:
                out=[]
                for item in h5f[key]:
                    out.append(item)
            return out
        return super().read(map_func,paras={"key":key,"is_attrs":is_attrs,"attrs":attrs})


    def save(self,asset_dict, attr_dict=None, mode="a"):
        def map_func(h5f,paras):
            asset_dict = paras["asset_dict"]
            attr_dict = paras["attr_dict"]
            for key, val in asset_dict.items():
                data_shape = val.shape
                if key not in h5f:
                    data_type = val.dtype
                    chunk_shape = (1, ) + data_shape[1:]
                    maxshape = (None, ) + data_shape[1:]
                    dset = h5f.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                    dset[:] = val
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                else:
                    dset = h5f[key]
                    dset.resize(len(dset) + data_shape[0], axis=0)
                    dset[-data_shape[0]:] = val      
            return {}         
        if mode =="a": super().add(map_func,paras={"asset_dict":asset_dict,"attr_dict":attr_dict})
        elif mode =="w": super().create(map_func,paras={"asset_dict":asset_dict,"attr_dict":attr_dict})



class patchImageStorage(h5Abstract):
    def __init__(self,loc:str) -> None:
        # handle file name
        self.loc = loc
        self.h5_handler = None
        # init data
        # slide patch img collection list of np.ndarray [(n,h,w,c)]
        self.patchs_list = None
        self.coords_list = None

        self.idx_ranges = None

        self.fnames = None
        self.folders = None

    def set_params(self,fnames:list,folders:list,
                        patchs_list:np.ndarray,coords_list:np.ndarray,
                        #new:bool=False
                        ) -> None:
        """
        set parameters for h5 file
        in:
            fname: list of file name(list of str ) each with length 64
            folder: list of folder name(list of str) each with length 36
            idx_range: list of idx range(list of tuple)
            patch: list of patch img( np.ndarray)
            coord: list of patch coord(np.ndarray)
        """
        #if new: self.clear_params()

        self.fnames = np.asarray(fnames).astype("<S64")
        self.folders = np.asarray(folders).astype("<S32")

        # set list data as numpy array
        # idx_ranges depend on previous data, new set as 0
        if self.idx_ranges is None: 
            idx_ranges=[0,patchs_list.shape[0]] 
        else: 
            idx_ranges=[self.idx_ranges[-1][-1],self.idx_ranges[-1][-1]+patchs_list.shape[0]]
        self.idx_ranges= np.asarray([idx_ranges])
        self.patchs_list = np.asarray(patchs_list)
        self.coords_list = np.asarray(coords_list)

    def clear_params(self) -> None:
        """
        clear parameters for new.update h5 file
        """
        self.fnames = None
        self.folders = None
        self.idx_ranges = None
        self.patchs_list = None
        self.coords_list = None

    def create(self) -> None:
        def map_func(h5f,paras):
            # get paras needed
            idx_ranges = paras["idx_ranges"]
            fnames = paras["fnames"]
            folders = paras["folders"]

            patchs_list = paras["patchs_list"]
            coords_list  = paras["coords_list"]

            # process logic
            h5f.create_group("data")
            h5f.create_group("meta")
            # data
            h5f.create_dataset("data/patchs_list", 
                                data=patchs_list,
                                maxshape=(None, patchs_list.shape[1],patchs_list.shape[2],patchs_list.shape[3]))
                                #compression="gzip")
            h5f.create_dataset("data/coords_list", 
                                data=coords_list,
                                maxshape=(None,coords_list.shape[1]))#compression="gzip")
            # meta
            h5f["meta"].create_dataset("fnames", data=fnames,maxshape=(None,))
            h5f["meta"].create_dataset("folders", data=folders,maxshape=(None,))
            h5f["meta"].create_dataset("idx_ranges", data=idx_ranges,maxshape=(None, idx_ranges.shape[1]))
            # return result
            return {}
        super().create(map_func,paras={"idx_ranges":self.idx_ranges,
                                        "patchs_list":self.patchs_list,
                                        "coords_list":self.coords_list,
                                        "fnames":self.fnames,
                                        "folders":self.folders})
    def release(self) -> None:
        """
        release h5 file
        """
        if self.h5_handler is not None:
            self.h5_handler.close()
            self.h5_handler = None

    def get_abs_patch_idx(self,fname:str,idx:int):
        """
        get absolute patch idx in h5 file
        in:
            fname: file name
            idx: patch idx within one slide file
        out:
            abs_idx: absolute patch idx
        """
        # get file idx
        #fname.encode("utf-8") in h5_worker.fnames
        file_idx = np.where(self.fnames == fname.encode("utf-8"))[0][0]
        # get absolute idx
        abs_idx = self.idx_ranges[file_idx,0] + idx
        return abs_idx

    def read(self,fname:str=None,patch_idx:int=None,is_new:bool=True) -> None:
        # TO-DO: Change behaviour to accelereate reading
        if is_new or self.h5_handler is None:
            # changed because of multiple-thread processing
            # https://github.com/pytorch/pytorch/issues/3415
            self.h5_handler = h5py.File(self.loc, "r",libver='latest', swmr=True)
            assert self.h5_handler.swmr_mode
            #
            self.fnames = self.h5_handler["meta/fnames"][:]
            self.folders = self.h5_handler["meta/folders"][:]
            self.idx_ranges = self.h5_handler["meta/idx_ranges"][:]
        
        if patch_idx is not None and fname is not None:
            # get patch index
            abs_idx = self.get_abs_patch_idx(fname,patch_idx)
            # get patch data
            patch = self.h5_handler["data/patchs_list"][abs_idx,...]
            coord = self.h5_handler["data/coords_list"][abs_idx,...]
            return coord,patch
        # if with self.release(),it allows multiple processing but supper slow
        #self.release()

    def add(self) -> None:
        def map_func(h5f,paras):
            # get paras needed
            assert paras["fnames"] not in h5f["meta"]["fnames"]
            # process logic
            for key, val in paras.items():
                if key == "patchs_list" or key == "coords_list":
                    h5f["data"][key].resize(len(h5f["data"][key]) + val.shape[0], axis=0)
                    h5f["data"][key][-val.shape[0]:] = val
                elif key == "idx_ranges":
                    h5f["meta"][key].resize(len(h5f["meta"][key]) + val.shape[0], axis=0)
                    h5f["meta"][key][-val.shape[0]:] = val
                elif key == "fnames" or key == "folders":
                    h5f["meta"][key].resize(len(h5f["meta"][key]) + val.shape[0], axis=0)
                    h5f["meta"][key][-val.shape[0]:] = val

            # return result
            return {}
        super().add(map_func,paras={"idx_ranges":self.idx_ranges,
                                        "patchs_list":self.patchs_list,
                                        "coords_list":self.coords_list,
                                        "fnames":self.fnames,
                                        "folders":self.folders})
"""
                    h5f["meta"][key].resize(len(h5f["meta"][key]) + val.shape[0], axis=0)
                    # for file patch mapping, need to link the patch id in the h5 and in slide
                    val = val + h5f["meta"][key].shape[0]
                    h5f["meta"][key][-val.shape[0]:] = val
                else:
"""




    

