"""
init function for pl models

"""
from HistoMIL.EXP.paras.trainer import PLTrainerParas
from HistoMIL.EXP.paras.dataset import DatasetParas
from HistoMIL.EXP.paras.optloss import OptLossParas

from HistoMIL.MODEL.Image.MIL.init import aviliable_mil_models
from HistoMIL.MODEL.Image.PL_protocol.MIL import pl_MIL
def create_img_model(train_paras:PLTrainerParas,
                     dataset_paras:DatasetParas,
                     optloss_paras:OptLossParas,
                     model_para):
    """
    create img pl model
    """
    #--------> get model paras
    model_name = train_paras.model_name
    #--------> create model
    if model_name in aviliable_mil_models.keys():
        if model_name == "ABMIL":
            from HistoMIL.MODEL.Image.MIL.ABMIL.pl import pl_ABMIL
            pl_model = pl_ABMIL(data_paras=dataset_paras,
                            opt_paras=optloss_paras,
                            trainer_paras=train_paras,
                            model_para=model_para)
        elif model_name == "TransMIL":
            from HistoMIL.MODEL.Image.MIL.TransMIL.pl import pl_TransMIL
            pl_model = pl_TransMIL(data_paras=dataset_paras,
                            opt_paras=optloss_paras,
                            trainer_paras=train_paras,
                            model_para=model_para)
        elif model_name == "DSMIL":
            from HistoMIL.MODEL.Image.MIL.DSMIL.pl import pl_DSMIL
            pl_model = pl_DSMIL(data_paras=dataset_paras,
                            opt_paras=optloss_paras,
                            trainer_paras=train_paras,
                            model_para=model_para)
    else:
        raise ValueError("model name not aviliable")
    return pl_model

def create_img_mode_paras(train_paras:PLTrainerParas,):
    #--------> get model paras
    model_name = train_paras.model_name
    #--------> create model
    if model_name in aviliable_mil_models.keys():
        if model_name == "ABMIL":
            from HistoMIL.MODEL.Image.MIL.ABMIL.paras import AttMILParas
            model_paras = AttMILParas()
        elif model_name == "TransMIL":
            from HistoMIL.MODEL.Image.MIL.TransMIL.paras import TransMILParas
            model_paras = TransMILParas()
        elif model_name == "DSMIL":
            from HistoMIL.MODEL.Image.MIL.DSMIL.paras import DSMILParas
            model_paras = DSMILParas()
    else:
        raise ValueError("model name not aviliable")

    return model_paras