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

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/detection/{}_without_family_below_{}".format(file_key, remove_family_below)
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
        def score(model):
            return indexes[:e]
        
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
models = list(models)

x = [20, 50, 80, 100, 150, 200, 250, 300, 350, 500]
matrix = torch.Tensor(matrix)

n_times = 10
results = {e: {"histogram_positive": [], "histogram_negative": [], "roc_TP": [], "roc_FP": []} for e in x}
for time_i in range(n_times):
    f_image_score = get_score(image_score_name)


    print("Time {}".format(time_i))
    data_roc = {e: {"infos": [], "labels": [], "models": []} for e in x}
    for f in list(families.keys()):
        #print("Family:", f)
        #print("Family:", families[f]["models"])
        # Reordered the images
        alice_model = get_delegate(families[f]["models"], families[f]["pure"])
        #alice_model = families[f]["pure"]
        alice_model_ind = models.index(alice_model)
        #print("Alice Model: {}".format(models[alice_model_ind]))

        images_indexes = f_image_score(families[f]["pure"])
        if image_score_name.startswith("random_wrong_"):    
            indexes_right = list(np.nonzero(matrix[:, alice_model_ind, 0].numpy() == truth)[0])
            indexes_wrong = list(np.nonzero(matrix[:, alice_model_ind, 0].numpy() != truth)[0])

            random.shuffle(indexes_right)
            random.shuffle(indexes_wrong)

        for n_images in x:
            if image_score_name.startswith("random_wrong_"):    
                n_wrong = int(n_images * float(image_score_name.split("_")[-1]))
                n_right = n_images - n_wrong
                images_indexes_n = indexes_wrong[:n_wrong] + indexes_right[:n_right]
            else:
                images_indexes_n = random.sample(images_indexes, n_images)

            v_o = matrix_gt_index[alice_model_ind, images_indexes_n].unsqueeze(0)
            v_o = v_o.repeat_interleave(len(models), 0)
            v_v = matrix_gt_index[:, images_indexes_n]

            p = list(model_distance(v_o, v_v).cpu().numpy())
            data_roc[n_images]["infos"] += p
            data_roc[n_images]["models"] += models

            for m_i, m in enumerate(models):
                data_roc[n_images]["labels"].append(m in families[f]["models"])
        


    for n_image in data_roc:
        print(Counter(data_roc[n_image]["labels"]))
        d = list(zip(data_roc[n_image]["labels"], data_roc[n_image]["infos"]))
        if distance_best == "min":
            d = sorted(d, key=lambda x: x[1], reverse=True)
        else:
            d = sorted(d, key=lambda x: x[1])
        length = len(d)

        n_tp = sum(data_roc[n_image]["labels"])
        n_fp = length - n_tp

        n_positive = n_tp
        n_neg = n_fp
        roc_fp, roc_tp = [], []
        roc_val = {}
        for l, inf in d:
            if not l:
                n_fp -= 1
            else:
                n_tp -= 1
            roc_fp.append(n_fp / n_neg)
            roc_tp.append(n_tp / n_positive)

        results[n_image]["roc_TP"].append(roc_tp)
        results[n_image]["roc_FP"].append(roc_fp)

        true_infos = []
        wrong_infos = []
        for i, aert in enumerate(data_roc[n_image]["labels"]):
            if aert:
                true_infos.append(data_roc[n_image]["infos"][i])
            else:
                wrong_infos.append(data_roc[n_image]["infos"][i])

        
        results[n_image]["histogram_positive"].append(true_infos)
        results[n_image]["histogram_negative"].append(wrong_infos)


results_mean = {e: [] for e in x}
for n_image in results:
    results_mean[n_image] = [
        np.mean(results[n_image]["roc_TP"], 0),
        np.std(results[n_image]["roc_TP"], 0),
        np.mean(results[n_image]["roc_FP"], 0),
        np.std(results[n_image]["roc_FP"], 0)
    ]

np.save(os.path.join(output_dir, output_filename + ".npy"), results_mean)

"""
plt.figure(figsize=(8, 5))
score_fp_setted = {}

roc_results = {}

for n_image in results:

    roc_tp = np.mean(results[n_image]["roc_TP"], 0)
    roc_fp = np.mean(results[n_image]["roc_FP"], 0)
    print("n_images", n_image)
    for i, e in enumerate(roc_fp.flatten()):
        if e < 0.05:
            print("FP: {} / TP: {}".format(e, roc_tp[i]))
            break
        
    std = np.std(results[n_image]["roc_TP"], 0)
    
    plt.plot(roc_fp, roc_tp, label="{} images".format(n_image))#, #color=colors[n_image])
    plt.fill_between(roc_fp, roc_tp - std, roc_tp + std, alpha=0.2) #, color=colors[n_image])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")


fig, axs = plt.subplots(1, len(results), figsize=(14, 5))
for i, n_image in enumerate(results):
    true_infos = np.mean(results[n_image]["histogram_positive"], 0)
    wrong_infos = np.mean(results[n_image]["histogram_negative"], 0)

    axs[i].hist(true_infos, bins=20, weights=np.ones(len(true_infos))/len(true_infos), label="True", color="g")
    if model_distance_name != "mutual_information":
        axs[i].set_xlim(0, 1)
    axs[i].hist(wrong_infos, bins=20, weights=np.ones(len(wrong_infos))/len(wrong_infos), label="False", color="r")
    axs[i].legend()
    axs[i].set_title(n_image)
plt.savefig(os.path.join(output_dir, output_filename + "_hist.pdf"), bbox_inches="tight")
"""