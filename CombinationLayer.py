import torch.nn as nn
import torch
import math
import torch.nn.functional as F
class CombinationLayer(nn.Module):
    def forward(self, query, key, value, dropout=None):
        query_key = query * key / math.sqrt(query.size(-1))
        query_value = query * value / math.sqrt(query.size(-1))
        tmpW = torch.stack([query_key, query_value], -1)
        tmpsum = torch.softmax(tmpW, dim=-1)
        tmpV = torch.stack([key, value], dim=-1)
        #print(tmpV.shape)
        #print(tmpsum.shape)
        tmpsum = tmpsum * tmpV
        tmpsum = torch.squeeze(torch.sum(tmpsum, dim=-1), -1)
        if dropout:
            tmpsum = dropout(tmpsum)
        return tmpsum


