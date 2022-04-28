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
    "--sort_images", type=str, default="random", 
    choices=["random", "probability", "diff_loss", "loss", "entropy_gt_index", "entropy_label"])
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images
model_distance_name = args.distance.lower()
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/detection_truth_from_delegate/{}".format(file_key)
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

models = np.load(models)
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
indexes_kept = []
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
        families[f]["indexes"].append(len(indexes_kept))
        indexes_kept.append(i)

models = new_models
originals_index = [i for i, m in enumerate(models) if get_original_and_variation(m)[1] is None]
matrix = matrix[:, indexes_kept]
print("Remaining models:", len(models))


####################
###########################
###########################
### Select Delegate


matrix_gt_index_for_deg = np.ones(matrix.shape[:2]) * top_k
for row_i, row in enumerate(matrix):
    t = truth[row_i]
    models_ind, truth_ind = np.where(row == t)
    matrix_gt_index_for_deg[row_i][models_ind] = truth_ind
matrix_gt_index_for_deg = matrix_gt_index_for_deg.astype(int)
matrix_gt_index_for_deg = torch.Tensor(matrix_gt_index_for_deg)


def get_delegate(fmodels, pure_model):
    m_i = models.index(pure_model)
    fmodels = sorted(fmodels)
    fm_i = [models.index(m) for m in fmodels]

    v_o = matrix_gt_index_for_deg[:, m_i].unsqueeze(0)
    v_o = v_o.repeat_interleave(len(fmodels), 0)
    v_v = matrix_gt_index_for_deg[:, fm_i].transpose(1, 0)
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
            return indexes[s:e]
        return score
        
    if score_name == "diff_loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                list(range(len(pure_model_infos[model]))),
                key=lambda i: pure_model_infos[model][i][0] - pure_model_infos[model][i][1]
            )
            s = int(len(indexes) * 0.15)
            e = int(len(indexes) * 0.4)
            return indexes[s: e]
        return score

    if score_name == "loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                list(range(len(pure_model_infos[model]))),
                key=lambda i: pure_model_infos[model][i][0]
            )
            s = int(len(indexes) * 0.05)
            e = int(len(indexes) * 0.3)
            return indexes[s:e]
        return score

    if score_name == "probability":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_probability.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                list(range(len(pure_model_infos[model]))),
                key=lambda i: pure_model_infos[model][i][0]
            )
            s = int(len(indexes) * 0.25)
            e = int(len(indexes) * 0.35)
            return indexes[s: e]
        return score
    if score_name == "random":
        indexes = list(range(len(matrix)))
        random.shuffle(indexes)
        def score(model):
            return indexes
        return score
    else:
        raise ValueError("Unknown Score:", score_name)

#####################################
#####################################


f_image_score = get_score(image_score_name)

models = list(models)

x = [50, 100]
data_roc = {e: {"infos": [], "labels": [], "models": []} for e in x}

matrix = torch.Tensor(matrix)
top_k_delegate = 1
for f in tqdm.tqdm(families):
    print("Family:", f)
    #print("Family:", families[f]["models"])
    # Reordered the images
    images_indexes = f_image_score(families[f]["pure"])
    #images_indexes = random.sample(images_indexes, max(x))
    images_indexes = images_indexes[:max(x)]


    alice_model = get_delegate(families[f]["models"], families[f]["pure"])
    alice_model_ind = models.index(alice_model)
    #print("Alice Model: {}".format(models[alice_model_ind]))
    print(alice_model)

    for n_images in x:
        tmp = torch.cat([matrix[images_indexes[:n_images]]] * (top_k_delegate + 1), 0)
        truth_f = list(truth[images_indexes[:n_images]])
        for top_i in range(top_k_delegate):
            truth_f += list(matrix[images_indexes[:n_images], alice_model_ind, top_i])
        truth_f = np.array(truth_f).reshape(-1)

        matrix_gt_index = np.ones(tmp.shape[:2]) * top_k
        for row_i, row in enumerate(tmp):
            t = truth_f[row_i]
            models_ind, truth_ind = np.where(row == t)
            matrix_gt_index[row_i][models_ind] = truth_ind
        matrix_gt_index = matrix_gt_index.astype(int)
        matrix_gt_index = torch.Tensor(matrix_gt_index)

        matrix_gt_index = matrix_gt_index.transpose(1, 0).long()
        v_o = matrix_gt_index[alice_model_ind, :].unsqueeze(0)
        v_o = v_o.repeat_interleave(len(models), 0)
        data_roc[n_images]["infos"] += list(model_distance(v_o, matrix_gt_index).cpu().numpy())
        data_roc[n_images]["models"] += models

        for m in models:
            data_roc[n_images]["labels"].append(m in families[f]["models"])
        


fig, axs = plt.subplots(1, len(data_roc), figsize=(14, 5))
for e_i, e in enumerate(data_roc):
    d = list(zip(data_roc[e]["labels"], data_roc[e]["infos"]))
    if distance_best == "min":
        d = sorted(d, key=lambda x: x[1], reverse=True)
    else:
        d = sorted(d, key=lambda x: x[1])

    true_infos = []
    wrong_infos = []
    for i, aert in enumerate(data_roc[e]["labels"]):
        if aert:
            true_infos.append(data_roc[e]["infos"][i])
        else:
            wrong_infos.append(data_roc[e]["infos"][i])
    axs[e_i].hist(true_infos, bins=20, weights=np.ones(len(true_infos))/len(true_infos), label="True", color="g")
    axs[e_i].set_xlim(0, 1)
    axs[e_i].hist(wrong_infos, bins=20, weights=np.ones(len(wrong_infos))/len(wrong_infos), label="False", color="r")
    axs[e_i].legend()
    axs[e_i].set_title(e)
plt.savefig(os.path.join(output_dir, output_filename + "_hist.pdf"), bbox_inches="tight")

plt.figure(figsize=(8, 5))
score_fp_setted = {}
roc_results = {}
for e in data_roc:
    d = list(zip(data_roc[e]["labels"], data_roc[e]["infos"]))
    if distance_best == "min":
        d = sorted(d, key=lambda x: x[1], reverse=True)
    else:
        d = sorted(d, key=lambda x: x[1])
    length = len(d)

    n_tp = sum(data_roc[e]["labels"])
    n_fp = length - n_tp

    n_positive = n_tp
    n_neg = n_fp
    roc_fp, roc_tp = [], []
    score_fp_setted[e] = {}
    roc_val = {}
    for l, inf in d:
        if not l:
            n_fp -= 1
        else:
            n_tp -= 1
        roc_fp.append(n_fp / n_neg)
        roc_tp.append(n_tp / n_positive)

        if n_fp / n_neg <= 0 and "0 - FP" not in roc_val:
            roc_val["0 - FP"] = n_fp / n_neg
            roc_val["0 - TP"] = n_tp / n_positive
        if n_fp / n_neg <= 0.01 and "0.01 - FP" not in roc_val:
            roc_val["0.01 - FP"] = n_fp / n_neg
            roc_val["0.01 - TP"] = n_tp / n_positive
        if n_fp / n_neg <= 0.05 and "0.05 - FP" not in roc_val:
            roc_val["0.05 - FP"] = n_fp / n_neg
            roc_val["0.05 - TP"] = n_tp / n_positive
        if n_fp / n_neg <= 0.1 and "0.1 - FP" not in roc_val:
            roc_val["0.1 - FP"] = n_fp / n_neg
            roc_val["0.1 - TP"] = n_tp / n_positive

    roc_results[e] = {"TP": roc_tp, "FP": roc_fp, "STEPS": roc_val}
    print("n_images", e)
    print(roc_val)
    
    plt.plot(roc_fp, roc_tp, label="{} images".format(e))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")
np.save(os.path.join(output_dir, output_filename + ".npy"), data_roc)

#np.save(os.path.join(output_dir, output_filename + ".npy"), results_to_save)
