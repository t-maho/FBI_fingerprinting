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
image_score_name = args.sort_images
model_distance_name = args.distance.lower()
remove_family_below = 1
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/identification/{}_without_family_below_{}".format(file_key, remove_family_below)
os.makedirs(output_dir, exist_ok=True)
output_filename = "stop_{}-{}-images_sorted_{}-delegate_{}".format(args.family, model_distance_name, image_score_name, args.delegate)


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

x = np.linspace(10, 400, 30)
#x = [10000]
x = [int(e) for e in x]
#x = [50, 100, 200]
results = {"successes": [], "fails": [], "unknown": [], "thresholds": []}

n_times = 10


mat_delegate = matrix_gt_index[delegate_models_indexes]
images_indexes = get_score(image_score_name)()
for time_i in range(n_times):
    highest_infos_labels = {n_images: [] for n_images in x}
    for m_i, m in enumerate(models):
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
                expectation = True
            highest_infos_labels[n_images].append((infos[best_i], labels[best_i], True not in labels))

    list_successes = np.zeros(len(x))
    list_unknown = np.zeros(len(x))
    list_fails = np.zeros(len(x))
    thresholds = []
    for i, n_images in enumerate(x):

        # Set a threshold to have FP = 5%

        if distance_best == "min":
            highest_infos_labels[n_images] = sorted(highest_infos_labels[n_images], key=lambda x: x[0], reverse=True)
        else:
            highest_infos_labels[n_images] = sorted(highest_infos_labels[n_images], key=lambda x: x[0])
        s = sum([l for _, l, _ in highest_infos_labels[n_images]])
        tp = sum([l for _, l, _ in highest_infos_labels[n_images]])
        unknowns = sum([is_unknown for _, _, is_unknown in highest_infos_labels[n_images]])
        fp = len(highest_infos_labels[n_images]) - tp - unknowns
        n_tp = tp
        n_fp = fp
        n_unknowns = unknowns

        for info, label in highest_infos_labels[n_images]:
            if label:
                tp -= 1
            else:
                fp -= 1
            if fp / n_fp < 0.05:
                list_successes[i] = tp / n_tp
                list_fails[i] = fp / n_fp
                list_unknown[i] = 1 - list_successes[i] - list_fails[i]
                thresholds.append(info)
                break

    results["list_successes"].append(list_successes)
    results["fails"].append(falist_failsils)
    results["unknown"].append(list_unknown)
    results["thresholds"].append(thresholds)

results_mean = {}
for k in results:
    results_mean[k + "_mean"] = np.mean(results[k], 0)
    results_mean[k + "_std"] = np.std(results[k], 0)
results_mean["x"] = x

np.save(os.path.join(output_dir, output_filename + ".npy"), results_mean)

width = 5
plt.figure(figsize=(10, 6))
plt.bar(x, results_mean["successes_mean"], width, color='g')
plt.bar(x, results_mean["unknown_mean"], width, bottom=results_mean["successes_mean"], color='grey')
plt.bar(x, results_mean["fails_mean"], width, bottom=results_mean["successes_mean"] + results_mean["unknown_mean"], color='r')
plt.xlabel("N Images")
plt.ylabel("Success Rate")
plt.legend(labels=['Success', 'Unknown', "Fail"])
plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")

