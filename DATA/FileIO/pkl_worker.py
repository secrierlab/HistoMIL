"""
basic io worker for pkl related part
"""
import pickle
import os
import io
import torch


class CPU_Unpickler(pickle.Unpickler):
    """
    get pkl data without consider devices
    https://stackoverflow.com/questions/57081727/load-pickle-file-obtained-from-gpu-to-cpu
    
    #contents = pickle.load(f) becomes...
    ontents = CPU_Unpickler(f).load()
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def save_pkl(filename, save_object)-> None:
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename)-> object:
    """
    load pkl file
    in:
        filename: str
    out:
        object: saved object
    """
    loader = open(filename,'rb')
    #file = pickle.load(loader)
    file = CPU_Unpickler(loader).load()
    loader.close()
    return file

def build_dir(path):
    """
    Checks if directory exists, if not, makes a new directory
    """
    if path and not os.path.exists(path):
        os.makedirs(path)