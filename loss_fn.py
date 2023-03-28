import torch.nn as nn
import torch

mse_loss = nn.MSELoss()
def mse_maskloss(output, target, mask):
    loss = [mse_loss(output[i][mask[i]==0], target[i][mask[i]==0]) \
            for i in range(len(output))]
    return torch.mean(torch.stack(loss))