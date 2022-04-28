import torch
import torch.nn as nn
import torch.nn.functional as F


class RSModel(nn.Module):
    def __init__(self, model, sigma=0.1, n_sample=100, seed=False):
        super(RSModel, self).__init__()
        self.model = model
        self.seed = seed
        self.sigma = sigma
        self.n_sample = n_sample

    def forward(self, x):
        if self.seed:
            torch.seed(self.seed)
        final_p = None
        loss= None
        for _ in range(self.n_sample):
            p = self.model(x + torch.randn_like(x) * self.sigma).detach()
            if loss == None:
                loss = p
            else:
                loss += p
            #if final_p is None:
            #    final_p = torch.zeros_like(p)
            #final_p += F.one_hot(torch.argmax(p, dim=1), num_classes=1000)
        return loss / self.n_sample
        #return final_p / self.n_sample

