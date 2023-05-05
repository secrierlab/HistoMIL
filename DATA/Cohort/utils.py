"""
tool functions 
"""
from pathlib import Path
import pandas as pd
from HistoMIL import logger
from HistoMIL.DATA.Slide.collector.data_collector import pre_process_wsi_collector

def cohort_pre_processing(df:pd.DataFrame,
                            data_locs,
                            collector_paras,                            
                            concepts:list=["slide","tissue","patch",],
                            fast_process:bool=True,force_calc:bool=False,):
    folders = df['folder'].values.tolist()
    fnames = df["filename"].values.tolist()
    for idx,(folder,fname) in enumerate(zip(folders,fnames)):
        if idx%100==0: logger.info(f"Cohort::pre-processing{idx}/{len(fnames)}..\n")
        wsi_loc = Path(str("/"+ folder +"/"+ fname))
        pre_process_wsi_collector(data_locs,
                                wsi_loc,
                                collector_paras=collector_paras,
                                concepts=concepts,
                                fast_process=fast_process,
                                force_calc=force_calc)