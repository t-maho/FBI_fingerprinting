import sys
from xml.parsers.expat import model
sys.path.append("/udd/tmaho/Projects/fingerprinting_real_images")
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import math
import argparse
from collections import Counter

from utils.model import get_original_and_variation
from utils.distances import get_model_distance


###########################
###########################
### Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--info", type=int, default=3)
parser.add_argument("--max_drop", type=int, default=-0.15)
parser.add_argument("--delegate", default="middle", choices=["far", "middle", "close"])
parser.add_argument("--small_merge", default="False")
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
parser.add_argument(
    "--sort_images", type=str, default="random")
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images
model_distance_name = args.distance.lower()
small_merge = args.small_merge.lower() in ["true", "y"]
family = "variation"
remove_family_below = 1
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

if small_merge:
    output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/identification_variation/{}_with_family_below_{}_merge".format(file_key, remove_family_below)
else:
    output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/identification_variation/{}_without_family_below_{}".format(file_key, remove_family_below)
os.makedirs(output_dir, exist_ok=True)
output_filename = "stop_{}-{}-images_sorted_{}-delegate_{}".format(family, model_distance_name, image_score_name, args.delegate)

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

    if small_merge:
        for k in models_gathered:
            families["SMALL from {}".format(k)] = {
                "models":  models_gathered[k],
                "pure": k
            }
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
    infos = list(model_distance(v_o, v_v).cpu().numpy())
    infos = [(i, inf) for i, inf in enumerate(infos)]
    infos = sorted(infos, key=lambda x: x[1])
    if args.delegate == "far":
        if distance_best == "max":
            return fmodels[infos[0][0]]
        else:
            return fmodels[infos[-1][0]]
    elif args.delegate == "close":
        if distance_best == "max":
            return fmodels[infos[-1][0]]
        else:
            return fmodels[infos[0][0]]
    elif args.delegate == "middle":
        i = int(len(fmodels) / 2)
        return fmodels[infos[i][0]]
    else:
        raise ValueError("Unknown delegate selection: {}".format(args.delegate))





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
    delegate_models.append(families[f]["delegate"])
    delegate_families.append(f)
delegate_models_indexes = [models.index(m) for m in delegate_models]

######################
######################
## Distance between each model and delegate

#x = np.linspace(20, 200, 15)
#x = [int(e) for e in x] + [250, 300, 350, 400, 500]
x = [1000, 2000, 3000, 4000]
#x = [50, 100, 200]

n_times = 10

images_indexes = get_score(image_score_name)()

if os.path.exists(os.path.join(output_dir, output_filename + ".npy")):
    highest_infos_labels_n_times = np.load(os.path.join(output_dir, output_filename + ".npy"), allow_pickle=True).item()
else:
    highest_infos_labels_n_times = {n_images: [] for n_images in x}

for time_i in range(n_times):
    print("time:", time_i)
    highest_infos_labels = {n_images: [] for n_images in x}
    for m_i, m in enumerate(models):
        delegate_families = []
        delegate_models = []
        m_o = get_original_and_variation(m)[0]
        for f in families:
            if families[f]["pure"] == m_o:
                delegate_models.append(families[f]["delegate"])
                delegate_families.append(f)
        delegate_models_indexes = [models.index(m) for m in delegate_models]
        mat_delegate = matrix_gt_index[delegate_models_indexes]

        m_v = matrix_gt_index[m_i].unsqueeze(0)
        m_v = m_v.repeat_interleave(len(delegate_models), 0)
        labels = [m in families[f]["models"] for f in delegate_families]

        for n_images in x:
            images_indexes_n = random.sample(images_indexes, n_images)
            infos = model_distance(mat_delegate[:, images_indexes_n], m_v[:, images_indexes_n])
            if distance_best == "max":
                best_i = infos.argmax()
            else:
                best_i = infos.argmin()
            if True not in labels:
                expectation = "unknown"
            else:
                expectation = delegate_families[labels.index(True)]
            highest_infos_labels[n_images].append((float(infos[best_i]), delegate_families[best_i], expectation))

    for n_images in x:
        highest_infos_labels_n_times[n_images].append(highest_infos_labels[n_images])

np.save(os.path.join(output_dir, output_filename + ".npy"), highest_infos_labels_n_times)

