"""
utils function to simplify the use files
"""
import numpy as np
from pathlib import Path

from HistoMIL.DATA.FileIO.h5_worker import patchCoordsStorage,patchImageStorage
from HistoMIL.DATA.FileIO.wsi_backends import select_backends

def location_to_fname(loc:str)->str:
    """
    convert location to fname
    in:
        loc: str
    out:
        folder:str
        fname: str
    """
    # read wsi data
    folder=str(Path(loc).parent).split("/")[-1]
    fname = Path(loc).name
    return folder,fname

def fname_to_PatientID(fname:str) -> str:
    """
    convert fname to patient id
    in:
        fname: str
    out:
        patient_id: str
    """
    return fname[:12]

def read_patch_coords(loc:str):
    """
    read patch coords from fname
    in:
        fname: str
    out:
        coords: list of tuple
        attr_dict: dict with attr ["patch_level","patch_size"]
    """
    h5worker = patchCoordsStorage(loc = loc)
    attr_dict = h5worker.read('coords',is_attrs=True,
                                            attrs=["patch_level","patch_size"])
    coords = h5worker.read('coords',is_attrs=False)
    return coords, attr_dict

def backends_collect_patch_to_numpy(wsi_loc:str,coords:list,attr_dict:dict)->np.ndarray:
    """
    collect patch to numpy by directly initate with select_backends
    in:
        wsi_loc: str:location of slide file
        coords: list of tuple: list of patch coord
        attr_dict: dict with attr ["patch_level","patch_size"]
    out:
        patch_data: np.ndarray with shape (n,h,w,c)
    """
    try:
        svs_worker = select_backends(Path(wsi_loc))
        svs_worker._open_handler_()
        patch_data = []
        # highlight here coords from previous step
        for i,coor in enumerate(coords):
            if i%1000==0: print(f"Collect patchs numpy {i}/{len(coords)}")
            patch = svs_worker.get_region(coords=(coor[0],coor[1]),
                                            patch_size=attr_dict["patch_size"],
                                            patch_level=attr_dict["patch_level"])
            # numpy add axis for channel dimension and append to list
            patch_data.append(np.expand_dims(patch,axis=0))
        patch_data = np.concatenate(patch_data,axis=0)
    except Exception as e: raise e
    finally:
        svs_worker._close_handler_()
    return patch_data

def get_idx_ranges(coords:list,current_ph5:patchImageStorage=None)->list:
    """
    get idx ranges
    in:
        coords: list of tuple: list of patch coord
        attr_dict: dict with attr ["patch_level","patch_size"]
    out:
        idx_ranges: list of tuple: list of idx range
    """
    # set start_idx 
    if current_ph5 is None:
        start_idx = 0
    else:
        current_ph5.read()
        start_idx =current_ph5.idx_ranges[-1][1] # the end_idx of last idx range
        #current_ph5.clear_params() # clear params for next use

    end_idx = start_idx+len(coords)
    idx_range = [start_idx,end_idx]

    return idx_range