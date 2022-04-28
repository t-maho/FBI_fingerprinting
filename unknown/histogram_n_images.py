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
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
parser.add_argument(
    "--sort_images", type=str, default="random")
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images
model_distance_name = args.distance.lower()
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/n_images_hitogram/{}".format(file_key)
os.makedirs(output_dir, exist_ok=True)
output_filename = "{}-images_sorted_{}".format(model_distance_name, image_score_name)


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

new_models = []
accuracies = {models[i]: (matrix[:, i, 0] == truth).sum(0) for i in range(len(models))}
for i, m in enumerate(models):
    o, m_v = get_original_and_variation(m)
    acc_drop = accuracies[m]  / accuracies[o] - 1
    if acc_drop > args.max_drop:
        new_models.append(m)

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
matrix_gt_index = torch.Tensor(matrix_gt_index).long().transpose(1, 0)


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
        def score(model):
            return indexes
        
    elif score_name == "diff_loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                list(range(len(matrix))),
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
                list(range(len(matrix))),
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
                list(range(len(matrix))),
                key=lambda i: pure_model_infos[model][i][0]
            )
            s = int(len(indexes) * 0.25)
            e = int(len(indexes) * 0.35)
            return indexes[s: e]

    elif score_name.startswith("random"):
        indexes = list(range(len(matrix)))
        random.shuffle(indexes)
        def score(model):
            return indexes
    else:
        raise ValueError("Unknown Score:", score_name)
    return score

#####################################
#####################################


f_image_score = get_score(image_score_name)
images_indexes = f_image_score("")
models = list(models)

x = [20, 50, 100, 250, 1000]
matrix = torch.Tensor(matrix)

results = {e: {"histogram_positive": [], "histogram_negative": [], 'infos': [], "labels": []} for e in x}

models_ov = [get_original_and_variation(m) for m in models]
for m_i, m in tqdm.tqdm(enumerate(models[:-1])):

    if image_score_name.startswith("random_wrong_"):    
        indexes_right = list(np.nonzero(matrix[:, m_i, 0].numpy() == truth)[0])
        indexes_wrong = list(np.nonzero(matrix[:, m_i, 0].numpy() != truth)[0])

        random.shuffle(indexes_right)
        random.shuffle(indexes_wrong)

    for n_images in x:
        if image_score_name.startswith("random_wrong_"):    
            n_wrong = int(n_images * float(image_score_name.split("_")[-1]))
            n_right = n_images - n_wrong
            images_indexes = indexes_wrong[:n_wrong] + indexes_right[:n_right]
        v_o = matrix_gt_index[m_i, images_indexes[:n_images]].unsqueeze(0)
        v_o = v_o.repeat_interleave(len(models) - m_i - 1, 0)
        v_v = matrix_gt_index[m_i + 1:, images_indexes[:n_images]]

        p = list(model_distance(v_o, v_v).cpu().numpy())
        results[n_images]["infos"] += p
        for m2_i, m2 in enumerate(models[m_i + 1:]):
            if models_ov[m_i] == models_ov[m_i + 1 + m2_i]:
                results[n_images]["labels"].append("Same Pure and Same Variation")
            elif models_ov[m_i][0] == models_ov[m_i + 1 + m2_i][0]:
                results[n_images]["labels"].append("Same Pure")
            else:
                results[n_images]["labels"].append("Different Pure")
    

np.save(os.path.join(output_dir, output_filename + ".npy"), results)



fig, axs = plt.subplots(1, len(results), figsize=(14, 5))
for i, n_image in enumerate(results):

    infos_sorted = {}
    for l_i, l in enumerate(results[n_image]["labels"]):
        if l not in infos_sorted:
            infos_sorted[l] = []
        infos_sorted[l].append(results[n_image]["infos"][l_i])

    labels = list(infos_sorted.keys())
    if model_distance_name != "mutual_information":
        axs[i].set_xlim(0, 1)
    axs[i].hist(
        [infos_sorted[l] for l in labels],
        bins=20,
        weights=[np.ones(len(infos_sorted[l]))/len(infos_sorted[l]) for l in labels],
        label=labels
    )
    axs[i].legend()
    axs[i].set_title(n_image)
plt.savefig(os.path.join(output_dir, output_filename + "_hist.pdf"), bbox_inches="tight")