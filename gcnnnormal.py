from torch import nn
import torch
from gelu import GELU
from SubLayerConnection import SublayerConnection
from Multihead_Combination import   MultiHeadedCombination
class GCNNM(nn.Module):
    def __init__(self, dmodel):
        super(GCNNM ,self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)
        self.comb = MultiHeadedCombination(8, dmodel)
        self.subconnect1 = SublayerConnection(dmodel, 0.1)
    def forward(self, state, inputad, rule):
        #print(rule.size())
        state = self.subconnect1(state, lambda _x:self.comb(_x, _x, rule, batch_size=1))#
        state = self.linear(state)
        #print(state.size())
        degree = torch.sum(inputad, dim=-1, keepdim=True).clamp(min=1e-6)
        degree2 = torch.sum(inputad, dim=-2, keepdim=True).clamp(min=1e-6)

        degree = 1.0 / torch.sqrt(degree)
        degree2 = 1.0 / torch.sqrt(degree2)
        degree2 = degree2 * inputad * degree 
        state2 = torch.matmul(degree2, state)
        #state = self.linearSecond(state)
        state = self.subconnect(state, lambda _x: self.com(_x, _x, state2, batch_size=1)) #state + torch.matmul(degree2, state)
        return state#self.dropout(state)[:,50:,:]

