import math

Settings_dicts={

    "steplr" :{"decay_step":10,"step_size":800,"sch_gamma":0.5},
    'explr':{"decay_step":10,"gamma":None,}, # for exp learning rate gamma need to calculate
    'multisteplr':{"milestones":None,"gamma":0.1},#"milestones" should be list of int

    #-------> for MAL_GNN_GAT model with gene profile matrix
    'linearlr':{"lr_lambda":None}, # use def_lambda_rule to create lambda function
    'plateaulr':{"mode":"min","factor":0.2,"threshold":0.01,"patience":5},

    'maxlr':{"T_max":None, "eta_min":0}, # need max_iter in T_max
}

def decay_epochs(max_iter:int,batch_size:int,data_len:int,ratio:float=0.67):
    decay_epochs = int(ratio * max_iter * batch_size / data_len) + 1
    return decay_epochs

def gamma_explr(decay_epochs):
    return math.exp(math.log(0.1) / decay_epochs)

def def_lambda_rule(num_epochs):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1) / float(num_epochs + 1)
        return lr_l
    return lambda_rule


