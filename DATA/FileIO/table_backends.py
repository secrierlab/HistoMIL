"""
This file is use to define CSV i/o related class for the project

(1) CSV IO to get a list of file loc and labels related

"""
##############################################################################
#             related import
###############################################################################
import os
import sys 
sys.path.append("..") 
sys.path.append(".") 
import pandas as pd
import random
from pathlib import Path
#############################################################################
#              Whole Slide Image Source
#############################################################################
class tableWorker:
    def __init__(self,loc:str) -> None:
        """
        tableWorker
        include file related functions for table data
        current support csv, xlsx, xls, Rdata
        """
        self.loc = loc
        self.file_type = Path(loc).suffix
        self.df = None

    def update_loc(self,loc:str):
        """
        update the loc of the table
        in:
            loc: str, the new loc of the table
        """
        self.loc = loc
        self.file_type = Path(loc).suffix

    def read_table(self,key:str=None) -> pd.DataFrame:
        """
        read a table from a csv xlsx xls Rdata file
        in:
            key: str,(optional) the key of the table(only for Rdata),if Rdata only include one object,key is None
        change:
            self.df: pd.DataFrame, the table read from the file
        """
        if self.file_type == ".csv":
            self.df = pd.read_csv(self.loc)
        elif self.file_type == ".xlsx" or self.file_type == ".xls":
            self.df =  pd.read_excel(self.loc)
        elif self.file_type == ".Rdata":
            import pyreadr
            result = pyreadr.read_r(self.loc)
            self.df = result[key]
        else:
            raise ValueError("unsupported file type")

    def write_table(self,df:pd.DataFrame=None,name:str=None) -> None:
        """
        write a table to a csv xlsx xls Rdata file
        in:
            df: pd.DataFrame, the table to write to the file
            key: str,(optional) the key of the table(only for Rdata),if Rdata only include one object,key is None
        """
        df = df if df is not None else self.df
        if self.file_type == ".csv":
            df.to_csv(self.loc)
        elif self.file_type == ".xlsx" or self.file_type == ".xls":
            df.to_excel(self.loc)
        elif self.file_type == ".Rdata":
            import pyreadr
            assert name is not None
            pyreadr.write_rdata(self.loc,df,df_name=name)
        elif self.file_type == "Rds":
            import pyreadr
            pyreadr.write_rds(self.loc,df)
        else:
            raise ValueError("unsupported file type")

    def show_csv_info(self,df:pd.DataFrame=None) -> None:
        """
        Get related info for a csv file
        in:
            df: pd.DataFrame, the table to get info
        """
        df = df if df is not None else self.df
        print("Keys of dataframe file is {}".format(df.keys()))
        print("There is {} items in csv file.".format(len(df)))
        print("The first 5 items in csv file is {}".format(df.head()))

    def df_to_list(self,keys:list,df:pd.DataFrame=None) -> list:
        """
        convert a dataframe to a list,each colume will be a list
        in:
            df: pd.DataFrame,(optional) the table to convert,if None,use self.df 
        out:
            list: list, the list of the dataframe
        """
        df = df if df is not None else self.df
        out = []
        for k in keys:
            out.append(df[k].values.tolist())
        return out




def shuffle(a,b):
    """
    shuffle two lists together
    in:
        a: list, the first list
        b: list, the second list
    """
    # shuffle two lists together
    assert len(a) == len(b)
    start_state = random.getstate()
    random.shuffle(a)
    random.setstate(start_state)
    random.shuffle(b)

