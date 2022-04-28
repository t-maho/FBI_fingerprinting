import torch.nn as nn

from torchvision import transforms as T


# re-scale and redistribute the values of the histogram of an image to broaden the covered range
class HistogramModel(nn.Module):
    def __init__(self, model, choice="equalize"):
        super(HistogramModel, self).__init__()
        if choice == "equalize":
            self._transform = lambda x: T.functional.equalize((x * 255).round().clip(0, 255).byte()) / 255
        else:
            raise ValueError("Unknown: {}".format(choice))
        self.model = model

    def forward(self, x):
        return self.model(self._transform(x))
