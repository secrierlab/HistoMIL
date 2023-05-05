"""
A simple class to manage locations

"""
import os
from HistoMIL import logger

class Locations:
    def __init__(self,root:str,sub_dirs:dict,is_build=False):
        self.root = root
        self.is_build = is_build
        self.is_exist(self.root)

        self.sub_dirs = sub_dirs


    def is_exist(self,loc):
        if not os.path.exists(loc):
            if self.is_build:
                os.makedirs(loc)
                logger.info(f" Built {loc}")
            else:
                raise ValueError(f"{loc} not exists..")

    def check_structure(self):
        for (key,_) in self.sub_dirs.items():
            loc = self.abs_loc(key)
            self.is_exist(loc)

    def abs_loc(self,name):
        """
        get abs location for a concept
        in:
            name::str::name of the concept
        out:
            ::str:: location string
        """
        return self.root + self.sub_dirs[name]


    

