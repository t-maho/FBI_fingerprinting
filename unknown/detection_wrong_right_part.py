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
parser.add_argument("--family", default="pure", choices=["pure", "variation"])
parser.add_argument("--delegate", default="far", choices=["far", "middle", "close"])
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
parser.add_argument(
    "--sort_images", type=str, default="random")
args = parser.parse_args()

top_k = args.info
model_distance_name = args.distance.lower()
image_score_name = args.sort_images
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/detection_right_wrong_part/{}".format(file_key)
os.makedirs(output_dir, exist_ok=True)
output_filename = "stop_{}-{}-images_sorted_{}".format(args.family, model_distance_name, image_score_name)

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
    if args.family == "pure":
        return o
    elif args.family == "variation":
        return o + "/" + m_v
    else:
        raise ValueError("Unknown family: {}".format(args.family))


new_models = []
accuracies = {models[i]: (matrix[:, i, 0] == truth).sum(0) for i in range(len(models))}
families = {}

tmp = [
    "FINETUNE-swin_tiny_patch4_window7_224-last_layer",
    "JPEG-swin_tiny_patch4_window7_224-60",
    "swin_tiny_patch4_window7_224",
    "FINETUNE-efficientnet_b0-last_layer",
    "JPEG-efficientnet_b0-60",
    "efficientnet_b0",
    "FINETUNE-resnet50-last_layer",
    "JPEG-resnet50-60",
    "resnet50"
]
for i, m in enumerate(models):
#for i, m in enumerate(tmp):
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
matrix_gt_index = torch.Tensor(matrix_gt_index).transpose(1, 0).long()


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
    f_base_indexes = lambda model: list(range(len(matrix)))

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
        def score(model):
            return indexes[s:e]
        
    elif score_name == "diff_loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                f_base_indexes(model),
                key=lambda i: pure_model_infos[model][i][0] - pure_model_infos[model][i][1]
            )
            s = int(len(indexes) * 0.15)
            e = int(len(indexes) * 0.4)
            return indexes[s: e]

    elif score_name == "loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                f_base_indexes(model),
                key=lambda i: pure_model_infos[model][i][0]
            )
            s = int(len(indexes) * 0.05)
            e = int(len(indexes) * 0.3)
            return indexes[s:e]

    elif score_name == "probability":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_probability.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                f_base_indexes(model),
                key=lambda i: pure_model_infos[model][i][0]
            )
            s = int(len(indexes) * 0.25)
            e = int(len(indexes) * 0.35)
            return indexes[s: e]

    elif score_name == "random":
        def score(model):
            indexes = f_base_indexes(model)
            random.shuffle(indexes)
            return indexes
    else:
        raise ValueError("Unknown Score:", score_name)
    return score

#####################################
#####################################

f_image_score = get_score(image_score_name)
models = list(models)

x = [50, 100, 200, 500]

matrix = torch.Tensor(matrix)


r = list(np.arange(0, 1, 0.05)) + [1]
#r = [0, 1]
x_to_str = lambda e: "{}% Positive / {}% Negative".format(e[0] * 100, e[1] * 100)
quotas = []
data_roc = {}
for e_r in r:
    e_r = np.round(e_r, 2)
    quotas.append((e_r, 1 - e_r))
    data_roc[x_to_str(quotas[-1])] = {e: {"infos": [], "labels": [], "models": []} for e in x}


for f in tqdm.tqdm(families):
    #print("Family:", f)
    #print("Family:", families[f]["models"])
    # Reordered the images

    #alice_model = get_delegate(families[f]["models"], families[f]["pure"])
    alice_model = families[f]["pure"]
    alice_model_ind = models.index(alice_model)


    images_indexes = f_image_score(families[f]["pure"])
    indexes_right = [i for i in images_indexes if matrix[i, alice_model_ind, 0] == truth[i]]
    indexes_wrong = [i for i in images_indexes if matrix[i, alice_model_ind, 0] != truth[i]]


    for n_images in x:
        for quota_right, quota_wrong in quotas:
            n_right = int(n_images * quota_right)
            n_wrong = int(n_images * quota_wrong)
            images_indexes = indexes_wrong[:n_wrong] + indexes_right[:n_right]

            v_o = matrix_gt_index[alice_model_ind, images_indexes].unsqueeze(0)
            v_o = v_o.repeat_interleave(len(models), 0)
            v_v = matrix_gt_index[:, images_indexes]

            p = list(model_distance(v_o, v_v, None).cpu().numpy())
            #print(p)
            data_roc[x_to_str((quota_right, quota_wrong))][n_images]["infos"] += p
            data_roc[x_to_str((quota_right, quota_wrong))][n_images]["models"] += models

            for m_i, m in enumerate(models):
                data_roc[x_to_str((quota_right, quota_wrong))][n_images]["labels"].append(m in families[f]["models"])
        



fig, axs = plt.subplots(1, len(x), figsize=(16, 5))
score_fp_setted = {}
roc_results = {}
for quota in data_roc:
    roc_results[quota] = {}
    for e_i, e in enumerate(data_roc[quota]):
        d = list(zip(data_roc[quota][e]["labels"], data_roc[quota][e]["infos"]))
        if distance_best == "min":
            d = sorted(d, key=lambda x: x[1], reverse=True)
        else:
            d = sorted(d, key=lambda x: x[1])
        length = len(d)

        n_tp = sum(data_roc[quota][e]["labels"])
        n_fp = length - n_tp

        n_positive = n_tp
        n_neg = n_fp
        roc_fp, roc_tp = [], []
        score_fp_setted[e] = {}
        for l, inf in d:
            if not l:
                n_fp -= 1
            else:
                n_tp -= 1
            roc_fp.append(n_fp / n_neg)
            roc_tp.append(n_tp / n_positive)

            if n_fp / n_neg <= 0.05 and e not in roc_results[quota]:
                roc_results[quota][e] = n_tp / n_positive

        axs[e_i].plot(roc_fp, roc_tp, label=quota)

for i, e in enumerate(x):
    axs[i].set_title(e)
    axs[i].legend()
    axs[i].set_xlabel("False Positive Rate")
    axs[i].set_ylabel("True Positive Rate")

plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")
np.save(os.path.join(output_dir, output_filename + ".npy"), data_roc)


import pandas as pd
df = pd.DataFrame.from_dict(roc_results, orient="index")
df.to_csv(os.path.join(output_dir, output_filename + ".csv"))