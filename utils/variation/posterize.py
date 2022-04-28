import torch.nn as nn
from torchvision import transforms as T


class PosterizeModel(nn.Module):
    def __init__(self, model, bits=8, probability=1):
        super(PosterizeModel, self).__init__()
        self.model = model
        self._transform_f = T.Compose([T.RandomPosterize(bits=bits, p=probability)])

    def forward(self, x):
        return self.model(self._transform(x))

    def _transform(self, x):
        x = (x * 255).round().clip(0, 255).byte()
        return self._transform_f(x) / 255

