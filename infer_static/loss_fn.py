import torch.nn as nn
import torch

ce_loss = nn.CrossEntropyLoss()
def ce_maskloss(output, target, mask):
    # The shape here: (1, 2) and (1), 2 is num of classes
    #taking mean to reduce the effect of the data length 
    loss = [ce_loss(output[i][mask[i]==0].mean(dim=-2).unsqueeze(0), target[i]) \
            for i in range(len(output))]
    return torch.mean(torch.stack(loss))