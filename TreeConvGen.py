from torch import nn
import torch
from gelu import GELU
class TreeConvGen(nn.Module):
    def __init__(self, kernel, dmodel):
        super(TreeConvGen ,self).__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(dmodel, dmodel, (1, kernel))
        self.activate = GELU()
    def forward(self, state, inputad, inputgen):
        tmp = []
        tmpState = state
        for i in range(self.kernel):
            tmpState = torch.matmul(inputad, tmpState)
            tmp.append(torch.matmul(inputgen, tmpState))
        states = torch.stack(tmp, 2)
        convstates = self.activate(self.conv(states.permute(0, 3, 1, 2)))
        convstates = convstates.squeeze(3).permute(0, 2, 1)
        return convstates