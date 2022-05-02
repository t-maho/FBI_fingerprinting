import torch
import os
import torchvision, timm

from utils.variation.histogram import HistogramModel
from utils.variation.pruning import PruneModel
from utils.variation.posterize import PosterizeModel
from utils.variation.precision import HalfPrecisionModel
from utils.variation.jpeg_compression import JPEGModel
from utils.variation.randomized_smoothing import RSModel



def get_original_and_variation_and_param(model_name):
    variation = None
    param = None
    if model_name.lower().startswith("prune"):
        _, model_name, layer, proportion = model_name.split("-")
        variation = "PRUNE ({})".format(layer)
        param = float(proportion)
    elif model_name.lower().startswith("half"):
        _, model_name = model_name.split("-")
        variation = "HALF"
    elif model_name.lower().startswith("posterize"):
        _, model_name, bits = model_name.split("-")
        variation = "POSTERIZE"
    elif model_name.lower().startswith("noisy"):
        _, model_name, norm = model_name.split("-")
        variation = "NOISY"
    elif model_name.lower().startswith("jpeg"):
        _, model_name, quality = model_name.split("-")
        variation = "JPEG"
        param = float(quality)
    elif model_name.lower().startswith("hist"):
        _, model_name, choice = model_name.split("-")
        variation = "HISTOGRAM"
    elif model_name.lower().startswith("rs"):
        _, model_name, sigma, n_sample = model_name.split("-")
        variation = "Randomized Smoothing"
        param = float(sigma)
    elif model_name.lower().startswith("finetune"):
        _, model_name, layers = model_name.split("-")
        variation = "FINETUNING ({})".format(layers.replace("_", " "))

    return model_name, variation, param

def get_original_and_variation(model_name):
    variation = None
    if model_name.lower().startswith("prune"):
        _, model_name, layer, proportion = model_name.split("-")
        variation = "PRUNE ({})".format(layer)
    elif model_name.lower().startswith("half"):
        _, model_name = model_name.split("-")
        variation = "HALF"
    elif model_name.lower().startswith("posterize"):
        _, model_name, bits = model_name.split("-")
        variation = "POSTERIZE"
    elif model_name.lower().startswith("noisy"):
        _, model_name, norm = model_name.split("-")
        variation = "NOISY"
    elif model_name.lower().startswith("jpeg"):
        _, model_name, quality = model_name.split("-")
        variation = "JPEG"
    elif model_name.lower().startswith("hist"):
        _, model_name, choice = model_name.split("-")
        variation = "HISTOGRAM"
    elif model_name.lower().startswith("rs"):
        _, model_name, sigma, n_sample = model_name.split("-")
        variation = "Randomized Smoothing"
    elif model_name.lower().startswith("finetune"):
        _, model_name, layers = model_name.split("-")
        variation = "FINETUNING ({})".format(layers.replace("_", " "))

    return model_name, variation


def get_model(model_name, jpeg_module=False, preload_model=None):
    extra_model_class = None
    use_before = False
    if model_name.lower().startswith("prune"):
        _, model_name, layer, proportion = model_name.split("-")
        proportion = float(proportion)
        if proportion > 1:
            proportion = int(proportion)

        extra_model_class=lambda x: PruneModel(x, layer_type=layer, proportion=proportion)
    elif model_name.lower().startswith("half"):
        _, model_name = model_name.split("-")
        extra_model_class = HalfPrecisionModel
    elif model_name.lower().startswith("posterize"):
        _, model_name, bits = model_name.split("-")
        extra_model_class = lambda x: PosterizeModel(x, bits=int(bits))
    elif model_name.lower().startswith("jpeg"):
        if jpeg_module:
            _, model_name, quality = model_name.split("-")
            extra_model_class = lambda x: JPEGModel(x, quality)
        else:
            _, model_name, quality = model_name.split("-")
            model_name = model_name
    elif model_name.lower().startswith("hist"):
        _, model_name, choice = model_name.split("-")
        extra_model_class = lambda x: HistogramModel(x, choice=choice)
    elif model_name.lower().startswith("rs"):
        _, model_name, sigma, n_sample = model_name.split("-")
        extra_model_class = lambda x: RSModel(x, sigma=float(sigma), n_sample=int(n_sample))
    elif model_name.lower().startswith("finetune"):
        use_before = True
        _, model_name, layers = model_name.split("-")
        folder = "/nfs/nas4/bbonnet/bbonnet/thibault/extra_model/model_retrained_from_val"
        filename = "{}-epoch_50-{}-lr_1e-05.pth".format(model_name, layers)
        if not os.path.exists(os.path.join(folder, filename)):
            raise ValueError("{} don't exist".format(filename))
        else:
            print(os.path.join(folder, filename))
            def extra_model_class(x):
                dict_update = {k[2:]: v for k, v in torch.load(os.path.join(folder, filename)).items()}
                x.load_state_dict(dict_update)
                return x

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    if preload_model:
        import copy
        normalizer, model = copy.deepcopy(preload_model)
    elif model_name == "madry":
        model = load_madry()
    elif model_name.lower().startswith("torchvision"):
        model = getattr(torchvision.models, model_name[len("torchvision_"):])(pretrained=True)
    else:
        model = timm.create_model(model_name, pretrained=True)
        mean = model.default_cfg["mean"]
        std = model.default_cfg["std"]
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)

    if extra_model_class is not None and use_before:
        model = extra_model_class(model)

    model = torch.nn.Sequential(
                    normalizer,
                    model
                )

    if extra_model_class is not None and not use_before:
        model = extra_model_class(model)

    if torch.cuda.is_available():
        model = model.cuda(0)
    model = model.eval()
    return model

def load_madry():
    import dill
    # load from https://download.pytorch.org/models/resnet50-19c8e357.pth
    weights_path = "/nfs/nas4/bbonnet/bbonnet/thibault/extra_model/imagenet_l2_3_0.pt"
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), pickle_module=dill)
    sd = checkpoint["model"]
    for w in ["module.", "attacker.", "model."]:
        sd = {k.replace(w, ""):v for k,v in sd.items()}

    std = sd["normalize.new_std"].flatten()
    mean = sd["normalize.new_mean"].flatten()
    
    del sd["normalize.new_std"]
    del sd["normalize.new_mean"]
    del sd["normalizer.new_std"]
    del sd["normalizer.new_mean"]

    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(sd)
    return model
