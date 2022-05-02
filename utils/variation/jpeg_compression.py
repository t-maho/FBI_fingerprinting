import torch.nn as nn

import torch
import numpy as np
from io import BytesIO
from PIL import Image


class JPEGModel(nn.Module):
    def __init__(self, model, quality=1):
        super(JPEGModel, self).__init__()
        self.quality = quality
        self.model = model

    def forward(self, x):
        return self.model(self._transform(x))

    def _transform(self, x):
        for i in range(len(x)):
            x[i] = self._jpeg_compression(x[i])
        return x

    def _jpeg_compression(self, x):
        pil_image = Image.fromarray((x * 255).detach().round().cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        f = BytesIO()
        pil_image.save(f, format='jpeg', quality=int(self.quality))
        jpeg_image = np.asarray(Image.open(f),).astype(np.float32).transpose(2, 0, 1) / 255.0
        jpeg_image = torch.Tensor(jpeg_image)
        if torch.cuda.is_available():
            jpeg_image = jpeg_image.to(0)
        
        return jpeg_image