# -*- coding: utf-8 -*-
#

import torch.nn as nn
import torchvision
import torch
from torch.nn import functional as F
import torchvision.models as models

class MyAlexnet(nn.Module):
    def __init__(self):
        super(MyAlexnet, self).__init__()
        self.base_model = models.alexnet(pretrained=True)
        self.base_model.classifier[-1]  = nn.Sequential()

    def forward(self, x):
        x = self.base_model(x)
        #print(('x4', x.shape), flush=True)
        return x

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class DataFusionModule(nn.Module):
    def __init__(self, centers):
        super(DataFusionModule, self).__init__()
        with torch.no_grad():
            tensor_centers = torch.tensor(centers, dtype=torch.float32)
        self.centers_layer = torch.nn.parameter.Parameter(tensor_centers)

    # normed query: B, D
    def forward(self, query):
        query = query.unsqueeze(1)
        value = F.normalize(self.centers_layer, p=2, dim=1)
        value = value.unsqueeze(0).repeat(query.size(0), 1, 1)
        
        batch_size, hidden_dim, centers_cnt = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(batch_size, centers_cnt), dim=1).unsqueeze(1)
        
        context = torch.bmm(attn, value).squeeze(1)
        return context #, attn

class AlexNetFc(nn.Module):
    def __init__(self, centers, args):
        super(AlexNetFc, self).__init__()
        
        self.myalexnet = MyAlexnet()
    
        self.datafusion_module = DataFusionModule(centers)

        self.hash_bit = args.hash_bit
        
        self.feats_pca_layer = nn.Sequential(nn.Linear(4096, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.feats_fusion_layer = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, self.hash_bit))

        self.hash_layer = nn.Sequential(nn.Tanh())
        

    def forward(self, x):
        x = self.myalexnet(x)
        x_pca_feats = self.feats_pca_layer(x)
        
        x_pca_feats = F.normalize(x_pca_feats, p=2, dim=1)
        centers_atten = self.datafusion_module(x_pca_feats)
        x_fused = torch.cat( (x_pca_feats, centers_atten), dim=1)

        x_fused = self.feats_fusion_layer(x_fused)

        x_hash = self.hash_layer(x_fused)
        
        return x_hash
