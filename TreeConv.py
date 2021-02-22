from torch import nn
import torch
from gelu import GELU
class TreeConv(nn.Module):
    def __init__(self, kernel, dmodel):
        super(TreeConv ,self).__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(dmodel, dmodel, (1, kernel))
        self.activate = GELU()
    def forward(self, state, inputad):
        tmp = [state]
        tmpState = state
        for i in range(self.kernel - 1):
            tmpState = torch.matmul(inputad, tmpState)
            tmp.append(tmpState)
        states = torch.stack(tmp, 2)
        convstates = self.activate(self.conv(states.permute(0, 3, 1, 2)))
        convstates = convstates.squeeze(3).permute(0, 2, 1)
        return convstates