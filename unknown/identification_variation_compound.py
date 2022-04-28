import copy
import sys
sys.path.append("/udd/tmaho/Projects/fingerprinting_real_images")
import tqdm
import os
import numpy as np
import random
import torch
import math
import argparse
from collections import Counter
from torch.utils.data import DataLoader
from datasets.dataset_test import CustomTestImageDataset

from utils.model import get_model, get_original_and_variation, get_original_and_variation_and_param
from utils.distances import get_model_distance


###########################
###########################
### Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--info", type=int, default=3)
parser.add_argument("--max_drop", type=int, default=-0.15)
parser.add_argument("--delegate", default="middle", choices=["far", "middle", "close"], nargs="+")
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
parser.add_argument(
    "--sort_images", type=str, default="random")
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images
model_distance_name = args.distance.lower()
family = "variation"

remove_family_below = 1
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/identification_variation_compound/{}_without_family_below_{}/delegate_{}".format(file_key, remove_family_below, "_".join(args.delegate))
os.makedirs(output_dir, exist_ok=True)
output_filename = "stop_{}-{}-images_sorted_{}".format(family, model_distance_name, image_score_name)


model_distance, distance_best = get_model_distance(model_distance_name)


###########################
###########################
### Load Data

# Params
matrix = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/matrix_{}.npy".format(file_key)

models = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/models.npy"
names = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/images.npy"
truth = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/test_ensemble_truth.npy"

models = list(np.load(models))
names = np.load(names)
matrix = np.load(matrix)
truth = np.load(truth, allow_pickle=True).item()
truth = np.array([truth[n] for n in names])
print(len(models))
print("matrix shape", matrix.shape)


if top_k == 1:
    matrix = np.expand_dims(matrix, -1)


###########################
###########################
### Retrieve original model labels

def get_family_name(m):
    o, m_v = get_original_and_variation(m)
    if m_v is None:
        m_v = "No Variation"
    if family == "pure":
        return o
    elif family == "variation":
        return o + "/" + m_v
    else:
        raise ValueError("Unknown family: {}".format(family))


new_models = []
accuracies = {models[i]: (matrix[:, i, 0] == truth).sum(0) for i in range(len(models))}
families = {}
for i, m in enumerate(models):
    o, m_v = get_original_and_variation(m)
    acc_drop = accuracies[m]  / accuracies[o] - 1
    if acc_drop > args.max_drop:
        f = get_family_name(m)
        if f not in families:
            families[f] = {"models": [], "indexes": [], "pure": o}
        new_models.append(m)
        families[f]["models"].append(m)

models_indexes = [models.index(m) for m in new_models]
models = new_models
originals_index = [i for i, m in enumerate(models) if get_original_and_variation(m)[1] is None]
originals = list(np.array(models)[originals_index])
matrix = matrix[:, models_indexes]


print("{} families.".format(len(families)))
if remove_family_below is not None and remove_family_below > 0:
    keys = list(families.keys())
    print("Removing families with less than {} members".format(remove_family_below))
    models_gathered = {}
    for f in keys:
        if len(families[f]["models"]) <= remove_family_below:
            if families[f]["pure"] not in models_gathered:
                models_gathered[families[f]["pure"]] = []
            models_gathered[families[f]["pure"]] += families[f]["models"]
            del families[f]

    print("{} families.".format(len(families)))

print("Remaining models:", len(models))
print("New Matrix Shape:", matrix.shape)


####################
###########################
###########################
### Select Delegate

matrix_gt_index = np.ones(matrix.shape[:2]) * top_k
for row_i, row in enumerate(matrix):
    t = truth[row_i]
    models_ind, truth_ind = np.where(row == t)
    matrix_gt_index[row_i][models_ind] = truth_ind
matrix_gt_index = matrix_gt_index.astype(int)
matrix_gt_index = torch.Tensor(matrix_gt_index).long().transpose(1, 0)


def get_delegate(fmodels, pure_model):
    m_i = models.index(pure_model)
    fmodels = sorted(fmodels)
    fm_i = [models.index(m) for m in fmodels]

    v_o = matrix_gt_index[m_i].unsqueeze(0)
    v_o = v_o.repeat_interleave(len(fmodels), 0)
    v_v = matrix_gt_index[fm_i]
    infos = list(model_distance(v_o.long(), v_v.long()).cpu().numpy())
    infos = [(i, inf) for i, inf in enumerate(infos)]
    infos = sorted(infos, key=lambda x: x[1])
    delegates = []
    for deleg in args.delegate:
        if deleg == "far":
            if distance_best == "max":
                delegates.append(fmodels[infos[0][0]])
            else:
                delegates.append(fmodels[infos[-1][0]])
        elif deleg == "close":
            if distance_best == "max":
                delegates.append(fmodels[infos[-1][0]])
            else:
                delegates.append(fmodels[infos[0][0]])
        elif deleg == "middle":
            i = int(len(fmodels) / 2)
            delegates.append(fmodels[infos[i][0]])
        else:
            raise ValueError("Unknown delegate selection: {}".format(args.delegate))
    return delegates





####################
###########################
###########################
### Score function to sort images

def get_score(score_name):
    if score_name == "entropy_label":
        scores = []
        for v in matrix[:, originals_index]:
            count = Counter([str(e) for e in v])
            count = [e / len(v) for e in count.values()]
            scores.append([- e * math.log(e, 2) for e in count if e != 0])
        indexes = sorted(
            list(range(len(matrix))),
            key=lambda i: scores[i],
            reverse=True
        )
        e = int(len(indexes) * 0.2)
        s = int(len(indexes) * 0)
        def score():
            return indexes[:e]
    elif score_name.startswith("random"):
        indexes = list(range(len(matrix)))
        random.shuffle(indexes)
        def score():
            return indexes
    else:
        raise ValueError("Unknown Score:", score_name)
    return score

#####################################
#####################################
## Get delegate  for each family

models = list(models)

delegate_families = []
delegate_models = []
for f in families:
    families[f]["delegate"] = get_delegate(families[f]["models"], families[f]["pure"])
    delegate_models += families[f]["delegate"]
    delegate_families += [f] * len(args.delegate)

delegate_models_indexes = [models.index(m) for m in delegate_models]

variation_min_max = {}
for m in models:
    m_o, m_v, param = get_original_and_variation_and_param(m)
    if param is None:
        continue
    if m_o not in variation_min_max:
        variation_min_max[m_o] = {}
    
    if m_v not in variation_min_max[m_o]:
        variation_min_max[m_o][m_v] = {"min": param, "max": param}
    variation_min_max[m_o][m_v]["min"] = min(param, variation_min_max[m_o][m_v]["min"])
    variation_min_max[m_o][m_v]["max"] = max(param, variation_min_max[m_o][m_v]["max"])

######################################
######################################
####### Select Compoung model



def get_prediction_model(images, truth, model_name, preload_model=None):
    model = get_model(model_name, jpeg_module=True, preload_model=preload_model)
    data_path = "/nfs/nas4/bbonnet/bbonnet/thibault/data/imagenet/test_224/"
    dataset = CustomTestImageDataset(data_path, order_from=images)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    for x, _ in data_loader:
        predictions.append(model(x.cuda(0) / 255).topk(top_k)[1].cpu())
    predictions = torch.cat(predictions, 0)
    #print(model_name)
    #print(predictions[:, 0])

    predictions_gt = np.ones(len(images)) * top_k
    for i, p in enumerate(predictions):
        if truth[i] in p:
            predictions_gt[i] = list(p).index(truth[i])
    return torch.Tensor(predictions_gt).long()

def get_random_variation(o, variation):
    if variation not in variation_min_max[o]:
        return None
    min_ = variation_min_max[o][variation]["min"] * 0.8
    max_ = variation_min_max[o][variation]["max"] * 1.2
    variation = variation.lower()
    if variation == "prune (conv)":
        r = random.uniform(min_, max_)
        return "PRUNE-{}-conv-{}".format(o, r)
    elif variation == "prune (ALL)":
        r = random.uniform(min_, max_)
        return "PRUNE-{}-all-{}".format(o, r)
    elif variation == "prune (last)":
        r = random.uniform(min_, min(1, max_))
        return "PRUNE-{}-last-{}".format(o, r)
    elif variation == "randomized smoothing":
        r = random.uniform(min_, min(1, max_))
        return "RS-{}-{}-100".format(o, r)
    elif variation == "jpeg":
        r = random.randint(min_, min(100, max_))
        return "JPEG-{}-{}".format(o, r)
    else:
        return None

######################
######################
## Distance between each model and delegate

#x = np.linspace(10, 300, 30)
#x = [int(e) for e in x]
x = [20,  50, 80, 100, 150, 200, 300, 400, 500]

n_times = 2
n_compound = 25
highest_infos_labels_n_times = {n_images: [] for n_images in x}
for time_i in range(n_times):
    images_indexes = get_score(image_score_name)()
    highest_infos_labels = {n_images: [] for n_images in x}
    images_indexes = random.sample(images_indexes, max(x))


    for o in originals:
        preload_model = get_model(o)

        delegate_families = []
        delegate_models = []
        for f in families:
            if o in families[f]["pure"]:
                delegate_models += families[f]["delegate"]
                delegate_families += [f] * len(families[f]["delegate"])
        delegate_models_indexes = [models.index(m) for m in delegate_models]
        mat_delegate = matrix_gt_index[delegate_models_indexes]

        # Unknown Predictions
        for m_i, m in enumerate(models):
            m_o = get_original_and_variation(m)[0]
            if m_o != o:
                continue

            labels = [m in families[f]["models"] for f in delegate_families]
            if True in labels:
                continue
            expectation = "unknown"

            m_v = matrix_gt_index[m_i].unsqueeze(0)
            m_v = m_v.repeat_interleave(len(delegate_models), 0)
            for n_images in x:
                images_indexes_n = random.sample(images_indexes, n_images)
                infos = model_distance(mat_delegate[:, images_indexes_n], m_v[:, images_indexes_n])
                highest_infos_labels[n_images].append((
                    infos.cpu().numpy(),
                    delegate_models,
                    m,
                    "unknown"))

        for compound_i in tqdm.tqdm(range(n_compound)):
            f_compound = random.choice(delegate_families)
            variation = f_compound.split("/")[1]

            #for fd_i, fd in enumerate(delegate_families):
            #    if variation == fd.split("/")[1]:
            #        print(delegate_models[fd_i])
            #        print(list(np.array(matrix[images_indexes, models.index(delegate_models[fd_i])]).flatten()))
            #        print(get_prediction_model(names[images_indexes], truth[images_indexes],  delegate_models[fd_i], preload_model=preload_model))
            #        print("\n")

            model_name = get_random_variation(o, variation)
            if model_name is None:
                continue

            m_v = get_prediction_model(names[images_indexes], truth[images_indexes], model_name, preload_model=preload_model)
            m_v = m_v.unsqueeze(0).repeat_interleave(len(delegate_models), 0)
            labels = [f == f_compound for f in delegate_families]

            for n_images in x:
                infos = list(model_distance(mat_delegate[:, images_indexes[:n_images]], m_v[:, :n_images]).cpu().numpy())
                highest_infos_labels[n_images].append((
                    infos,
                    delegate_models,
                    model_name,
                    "in"))
        
        np.save(os.path.join(output_dir, "time_{}-".format(time_i) + output_filename + ".npy"), highest_infos_labels)
    for n_images in x:
        highest_infos_labels_n_times[n_images].append(highest_infos_labels[n_images])

    np.save(os.path.join(output_dir, output_filename + ".npy"), highest_infos_labels_n_times)

