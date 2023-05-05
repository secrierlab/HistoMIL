"""
Define basic item for concept and collector for conceptsß
"""


def str_out(out:str,is_print:bool=False):
    if is_print:
        print(out)


class Items(object):
    def pre_requests(self):
        raise NotImplementedError
    
    def loc(self):
        raise NotImplementedError

    def calc(self):
        raise NotImplementedError
    
    def read(self):
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

class DataCollector(object):
    data_dict={}

    def in_dict(self,item:str):
        raise NotImplementedError

    def create(self,item:str):
        raise NotImplementedError

    def read(self,item:str):
        raise NotImplementedError

    def update(self,item:str,**kwargs):
        raise NotImplementedError

    def item_loc(self,item:str):
        raise NotImplementedError

    def release(self,item:str=None):
        raise NotImplementedError