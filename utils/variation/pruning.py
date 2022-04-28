from torch.nn.utils import prune
import torch.nn as nn


class PruneModel(nn.Module):
    def __init__(self, model, layer_type="conv", proportion=0.5):
        super(PruneModel, self).__init__()
        self.proportion = proportion
        self.model = self.prune_model_l1_unstructured(model, layer_type)

    def forward(self, x):
        return self.model(x)

    def prune_model_l1_unstructured(self, model, layer_type="conv"):
        if layer_type.lower().strip() == "all":
            for module in model.modules():
                if hasattr(module, "weight") and module.weight is not None:
                    prune.l1_unstructured(module, 'weight', self.proportion)
                    prune.remove(module, 'weight')
                if hasattr(module, "bias")and module.bias is not None:
                    prune.l1_unstructured(module, 'bias', self.proportion)
                    prune.remove(module, 'bias')
        elif layer_type.lower().strip() == "conv":
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, 'weight', self.proportion)
                    prune.remove(module, 'weight')
        elif layer_type.lower().strip() == "last":
            for m in list(model.modules())[::-1]:
                if isinstance(m, nn.Linear):
                    prune.l1_unstructured(m, 'weight', self.proportion)
                    prune.l1_unstructured(m, 'bias', self.proportion)
                    break
        elif layer_type.lower().strip() == "last_5":
            for m in list(model.modules())[-5:]:
                if isinstance(m, nn.Linear):
                    prune.l1_unstructured(m, 'weight', self.proportion)
                    prune.l1_unstructured(m, 'bias', self.proportion)
        else:
            raise ValueError("UnknownLayer {}".format(layer_type))

        return model