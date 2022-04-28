import torch.nn as nn


class HalfPrecisionModel(nn.Module):
    def __init__(self, model):
        super(HalfPrecisionModel, self).__init__()
        self.model = self._half(model)

    def forward(self, x):
        return self.model(x.half())

    def _half(self, model):
        model = model.half()
        return model
