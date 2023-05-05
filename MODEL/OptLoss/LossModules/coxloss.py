import torch
import numpy as np
import torch.nn as nn

def CoxLoss(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    # print("R mat shape:", R_mat.shape)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    # print("censor and theta shape:", censor.shape, theta.shape)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def reg_loss(model):
    for W in model.parameters():
        loss_reg = torch.abs(W).sum()
    return loss_reg

def Mixed_CoxLoss_with_reg(
                        # for survival loss
                        surv_batch_labels,censor_batch_labels,
                        # grad prediction and labels
                        grad_batch_labels,preds,
                        # model for loss reg
                        model,
                        #-----------> parameters
                        lambda_cox,lambda_nll,lambda_reg):
    loss_cox = CoxLoss(surv_batch_labels, censor_batch_labels, preds) 
    loss_reg = reg_loss(model)
    loss_func = nn.CrossEntropyLoss()
    grad_loss = loss_func(preds, grad_batch_labels)
    loss = lambda_cox * loss_cox + lambda_nll * grad_loss + lambda_reg * loss_reg
    return loss

def CELoss_with_reg(model,pred,label,
                    lambda_nll,lambda_reg):
    loss_reg = reg_loss(model)
    loss_func = nn.CrossEntropyLoss()
    loss_grad = loss_func(pred, label)
    loss = lambda_nll * loss_grad + lambda_reg * loss_reg
    return loss