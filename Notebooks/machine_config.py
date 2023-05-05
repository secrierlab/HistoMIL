"""
config local machine and user with all files of a cohort
"""

localcohort_name = "COAD"
#--------------------------> init machine and person
from HistoMIL.DATA.Cohort.location import Locations
data_locs = Locations(  root=f"/Path/to/file/{localcohort_name}/",
                                sub_dirs={
                                        "slide":f"TCGA-{localcohort_name}/",
                                        "tissue":"Tissue/",
                                        "patch":"Patch/",
                                        "patch_img":"Patch_Image/",# need to add for SSL
                                        "feature":"Feature/",
                                        })
exp_locs = Locations(  root="/Path/to/experiment/folder/",
                                sub_dirs={
                                        "src":"HistoMIL/",

                                        "idx":"/Data/",

                                        "saved_models":"/SavedModels/",
                                        "out_files":"/OutFiles/",

                                        "temp":"/Temp/",
                                        "user":"/User/",
                                        
                                     })

from HistoMIL.EXP.workspace.env import Machine
machine = Machine(data_locs,exp_locs)
from HistoMIL.EXP.workspace.env import Person
user = Person(id="0001")
user.name = "your user name"
user.wandb_api_key = "your wandb api key"

# save as pickle
import pickle
loc = exp_locs.abs_loc("user")
with open(f"/{loc}/{localcohort_name}_machine_config.pkl", 'wb') as f:
    pickle.dump([data_locs,exp_locs,machine,user], f)