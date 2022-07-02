import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import gc
from copy import deepcopy

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss
    

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()

# dist_loss = opts.dist_ratio * dist_criterion(e, t_e)
# angle_loss = opts.angle_ratio * angle_criterion(e, t_e)

class FewShotNCALossMultiLabel(torch.nn.Module):
    def __init__(
        self,
        classes=14,
        temperature=1,
        batch_size=16,
        num_labels=14,
        frac_negative_samples=1,
        frac_positive_samples=1,
    ):
        super(FewShotNCALossMultiLabel, self).__init__()
        self.temperature = torch.tensor(float(temperature), requires_grad=True).to(device)
        self.num_labels=num_labels

    def forward(self, pred, target):
        n, d = pred.shape

        p_norm = torch.pow(torch.cdist(pred, pred), 2)
        p_norm[p_norm < 1e-10] = 1e-10
        dist = torch.exp(-1 * p_norm / self.temperature).to(device)        

        all_pos = torch.zeros(n,n).to(device)
        all_negs = torch.zeros(n,n).to(device)
        negated_target = torch.tensor(~(target.type(torch.bool)),  dtype=torch.int16)

        for i in range(self.num_labels):
            positives = (target[:, i, None]*target[:, i, None].T)
            all_pos+= positives

        negatives_matrix = torch.tensor(~all_pos.type(torch.bool), dtype=torch.int16).to(device)

        positives_matrix = (
                    torch.tensor(all_pos, dtype=torch.int16)
                ).to(device)

        positives_matrix.fill_diagonal_(0)

        numerators = torch.sum(dist * positives_matrix, axis=0)
        denominators = torch.sum(dist* negatives_matrix, axis=0)

        denominators[denominators < 1e-10] = 1e-10
        frac = numerators / (numerators + denominators )

        loss = -1 * torch.sum(torch.log(frac[frac >= 1e-10])) / (n)   
        return loss