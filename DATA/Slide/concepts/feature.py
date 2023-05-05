"""
A basic part for feature embedding generation and usage
"""
#############################################################################
#               import 
#############################################################################
import numpy as np
from pathlib import Path
import torch
import os
from HistoMIL import logger
import timm

import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import Dataset,DataLoader


#############################################################################
#               import within package
#############################################################################
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.DATA.Slide.collector.items  import Items

from HistoMIL.EXP.paras.slides import FeatureParas
#from histocore.DATA.FileIO.pkl_worker import save_pkl,load_pkl

from HistoMIL.DATA.Slide.concepts.WholeSlideImage import WholeSlideImage
from HistoMIL.DATA.Slide.concepts.patch import Patches

#from histocore.DATA.Slide.concepts.nuclei import Nuclei
from HistoMIL.DATA.Database.data_aug import no_transforms
#############################################################################
#               class for graph related
#############################################################################
class Features(Items):
    def __init__(self,db_loc:Locations,wsi_loc:str,paras:FeatureParas) -> None:
        # 
        logger.debug("Feature:: Init instance for Feature embedding vector ")
        self.db_loc = db_loc
        self.paras = paras
        self.wsi_loc = Path(wsi_loc)
        self.model_name = paras.model_name
        self.feats_loc = None

        self.feature_embedding = None #torch.Tensor [N x dim] or [N x dim1 x dim2]
        self.feature_cluster = None   # np.array    [N x 1] or [N x m] with m clusters

        self.slide_size = None


    def pre_requests(self):
        return ["slide","patch"]
    
    def loc(self):
        assert self.model_name is not None
        folder = str(self.wsi_loc.parent).split("/")[-1]
        fname  = self.wsi_loc.name
        return Path(self.db_loc.abs_loc("feature")+f"/{self.model_name}/"+folder+"."+fname+".pt")

    def calc(self,
             slide:WholeSlideImage,
             patches:Patches,
             paras:FeatureParas,
             ):
        logger.debug("Feature:: Calculating feature vector.")
        slide.read()
        self.slide_size = slide.meta.shape

        self.extractor = Features_extractor(paras=paras)
        
        self.extractor._init_model()

        self.extractor.process(slide=slide,patches=patches)

        self.feature_embedding = self.extractor.feats
        self.save()
    
    def read(self,paras:FeatureParas=None):
        assert ((paras is not None) or (self.model_name is not None))
        self.feats_loc = self.loc()
        logger.debug(f"Feature:: Read feature embedding vector from {self.feats_loc}")
        self.feature_embedding = torch.load(self.feats_loc)
        if self.feature_embedding is None:
            logger.debug(f"Feature:: In {self.feats_loc} feature vector is empty")

    def len(self):
        return self.feature_embedding.shape[0]

    def save(self):
        torch.save(self.feature_embedding,self.feats_loc)
        logger.debug(f"Feature:: Feature embedding saved at {self.feats_loc}")

    def get(self,slide:WholeSlideImage,
             patches:Patches,paras:FeatureParas=None,force_calc:bool=False):
    
        self.paras = paras if paras is not None else self.paras
        #req_dict
        self.model_name = self.paras.model_name
        self.feats_loc = self.loc()
        if os.path.isfile(self.feats_loc) and not force_calc:
            self.read()
        else:
            self.calc(slide=slide,
                      patches=patches,
                      paras = self.paras)

    def get_datapoint(self,idx:int):
        assert self.feature_embedding is not None
        return self.feature_embedding[idx,:]

    def save_cluster(self,cluster=None,semantic_shifts=None):
        if cluster is not None:
            self.feature_cluster = cluster
        else:
            assert self.feature_cluster is not None
        cluster_loc = self.feats_loc.replace(".pt",f"_cluster_{self.paras.cluster_nb}.npy")
        np.save(cluster_loc,self.feature_cluster)
        if semantic_shifts is not None:
            sem_loc = self.feats_loc.replace(".pt",f"_semantic_shifts_{self.paras.cluster_nb}.npy")
            np.save(sem_loc,semantic_shifts)

    def read_cluster(self):
        cluster_loc = self.feats_loc.replace(".pt",f"_cluster_{self.paras.cluster_nb}.npy")
        self.feature_cluster = np.load(cluster_loc)
        if self.paras.with_semantic_shifts:
            sem_loc = self.feats_loc.replace(".pt",f"_semantic_shifts_{self.paras.cluster_nb}.npy")
            self.semantic_shifts = np.load(sem_loc)

class Features_extractor:
    def __init__(self,paras:FeatureParas) -> None:
        logger.debug("Feature:: Init feature extractor.")
        self.paras = paras
        self.device = paras.device
        self.trans = paras.trans
        self.model_name = paras.model_name


        self.feats = None
        self.c_label = None
        self.supported_model_list = timm.list_models(pretrained=True)

    def process(self,slide:WholeSlideImage,patches:Patches):
        # with pytorch dataloader
        with torch.no_grad():
            feats=[]
            self.get_dataloader(slide=slide,
                                patches=patches,
                                )
            for i, x in enumerate(self.dataloader, 0):
                if i%100==0: logger.info(f"{i}/{len(self.dataloader)}")
                x = x.to(self.device)
                f = self.model(x)
                f = f.view(f.shape[0],-1).detach().cpu() # B x N where N=CxWxH
                feats.append(f)
            # release source
            self.dataloader.dataset.wsi.close()
            self.feats = torch.cat(feats,dim=0)

    def _init_model(self):
        logger.debug(f"Feature:: Init feature extractor model{self.model_name}")
        model_instance = self.paras.model_instance#["model_instance"]
        if model_instance is not None:
            # init infer model from a customised model
            assert self.paras.img_size is not None and self.paras.out_dim is not None
            self.model = model_instance.to(self.device)
            self.img_size = self.paras.img_size
            self.out_dim  = self.paras.out_dim
        else:
            # init model from pretrained timm
            assert self.model_name in self.supported_model_list
            logger.debug("Feature:: Building pre-trained part from timm pkg.")
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            self.model.to(self.device)

            self.img_size = self.paras.img_size
            self.out_dim  = self.paras.out_dim
            if self.img_size is None or self.out_dim is None:
                self.img_size,self.out_dim = self._model_dims()


    def _model_dims(self):

        self.model.eval()
        # get 
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)

        # get a input data
        url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
        urllib.request.urlretrieve(url, filename)
        img = Image.open(filename).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device) # transform and add batch dimension

        with torch.no_grad():
            out = self.model(img_tensor)
            out = out.view(out.shape[0],-1)
        in_size = (config["input_size"][1],config["input_size"][2])
        out_dim = out.shape[1]

        return in_size,out_dim


    def get_dataloader(self,slide:WholeSlideImage,patches:Patches):
        # get a data loader for feat calc
        dataset = Featset(wsi = slide,
                          patches = patches,
                          img_size=self.img_size,
                          trans=self.trans,
                          is_train=False)

        self.dataloader = DataLoader(dataset, 
                                     batch_size=self.paras.batch_size,
                                     shuffle=False)

    def fit_cluster(self,):
        from sklearn.cluster import KMeans
        self.cluster = KMeans(n_clusters=self.paras.cluster_nb)
        data = self.feats.cpu().detach().numpy()
        self.cluster.fit(data)
        self.c_label=torch.from_numpy(self.cluster.predict(data))
    
    def get_cluster_center(self)->np.ndarray:
        return torch.from_numpy(self.cluster.cluster_centers_)

    def get_representative(self):
        # get dists
        dists = torch.empty((0, self.paras.cluster_nb)).to(self.device)
        centers = self.get_cluster_center()
        for i in range(self.feats.shape[0]):
            sample = self.feats[i,...]
            dist = torch.sum(torch.mul(sample - centers, sample - centers), (1))
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        # argmin
        return torch.argmin(dists, (0))


#################################################################################
#   define a dataset for feature extraction
################################################################################
# for some use case, we only want read the image once and calc features

class Featset(Dataset):
    def __init__(self, wsi:WholeSlideImage,patches:Patches,
                        img_size=None,trans=no_transforms,is_train=True):
        # patch_list include [collector,idx,label]
        self.wsi = wsi
        self.wsi.read()

        patches.read()
        assert patches.coords_dict is not None
        self.patches_list = patches.coords_dict
        self.patch_size = patches.patch_size
        self.patch_level = patches.patch_level

        self.img_size = img_size

        self.trans = trans(is_train=is_train) if trans is not None else None
        self.is_train = is_train


    def __len__(self):
        return len(self.patches_list)

    def _processing(self,img):
        if self.img_size is not None:
            pil_img=Image.fromarray(img)
            pil_img = pil_img.resize(self.img_size)
            img = np.asarray(pil_img)
        # get transform
        if self.trans is not None:
            img = self.trans(img)#,is_train=self.is_train)#.unsqueeze(0)
        return img

    def __getitem__(self, idx):
        
        coords = self.patches_list[idx]
        img = self.wsi.get_region(coords = coords,
                                 patch_size = self.patch_size,
                                  patch_level = self.patch_level)

        img = self._processing(img)
        return img
