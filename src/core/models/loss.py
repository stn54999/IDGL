import torch
import numpy as np
from lifelines.utils import concordance_index

def CoxLoss(hazard_pred=None,labels=None,device=None):
    survtime=labels[:,0]
    censor=labels[:,1]
    current_batch_len = len(survtime)
    #DSLoss=DeepSurvLoss(hazard_pred,survtime,censor)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]
    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox
#concordance_index(true[:, 0], -pred, true[:, 1])