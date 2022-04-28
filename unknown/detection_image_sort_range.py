import sys
sys.path.append("/udd/tmaho/Projects/fingerprinting_real_images")

import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from collections import Counter
import argparse
import random


from utils.distances import get_model_distance
from utils.model import get_original_and_variation

###########################
###########################
### Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--info", type=int, default=3)
parser.add_argument("--n_images", type=int, default=200)
parser.add_argument("--family", default="pure", choices=["pure", "variation", "singleton"])
parser.add_argument("--delegate", default="close", choices=["close", "far", "middle"])
parser.add_argument("--max_drop", type=float, default=-0.15)
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
parser.add_argument(
    "--sort_images", type=str, default="probability")
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images if args.sort_images.lower() not in ["none"] else None
model_distance_name = args.distance.lower()

if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/very_limited_information/detection_which_part/{}".format(file_key)
os.makedirs(output_dir, exist_ok=True)
if image_score_name is not None:
    output_filename = "{}-images_sorted_{}".format(model_distance_name, image_score_name)
else:
    output_filename = "{}".format(model_distance_name)



###########################
###########################
### Score function to sort models

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
print(len(models))
print("matrix shape", matrix.shape)

truth = np.array([truth[n] for n in names])

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
    elif args.family == "singleton":
        return m
    else:
        raise ValueError("Unknown family: {}".format(args.family))

accuracies = {models[i]: (matrix[:, i, 0] == truth).sum(0) for i in range(len(models))}
families = {}
models_kept = []
for i, m in enumerate(models):
    o, m_v = get_original_and_variation(m)
    acc_drop = accuracies[m]  / accuracies[o] - 1
    if acc_drop > args.max_drop:
        f = get_family_name(m)
        if f not in families:
            families[f] = {"models": [], "pure": o}
        families[f]["models"].append(m)
        models_kept.append(m)

models_indexes = [models.index(m) for m in models_kept]
models = models_kept
matrix = matrix[:, models_indexes]
originals_index = [i for i, m in enumerate(models) if get_original_and_variation(m)[1] is None]
print("New matrix shape", matrix.shape)

###########################
###########################
###########################
### Matrix Ground Truth Index


matrix_gt_index = np.ones(matrix.shape[:2]) * top_k
for row_i, row in tqdm.tqdm(enumerate(matrix)):
    t = truth[row_i]
    models_ind, truth_ind = np.where(row == t)
    matrix_gt_index[row_i][models_ind] = truth_ind
matrix_gt_index = matrix_gt_index.astype(int)
matrix_gt_index = torch.Tensor(matrix_gt_index)


####################
###########################
###########################
### Select Delegate

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


###########################
###########################
### Score function to sort images

def get_score(score_name):
    f_base_indexes = lambda model: list(range(len(matrix)))
    if len(score_name.split("-")) == 2:
        if score_name.split("-")[1] == "wrong":
            def f_base_indexes(model):
                index = models.index(model)
                m = matrix[:, index, 0]
                return [i for i, e in enumerate(m) if e != truth[i]]
        elif score_name.split("-")[1] == "right":
            def f_base_indexes(model):
                index = models.index(model)
                m = matrix[:, index, 0]
                return [i for i, e in enumerate(m) if e == truth[i]]
        else:
            raise ValueError
        score_name = score_name.split("-")[0]

    if score_name == "entropy_label":
        scores = []
        for v in matrix[:, originals_index]:
            count = Counter([str(e) for e in v])
            count = [e / len(v) for e in count.values()]
            scores.append([- e * math.log(e, 2) for e in count if e != 0])
        indexes = sorted(
            list(range(len(matrix_gt_index))),
            key=lambda i: scores[i]
        )
        def score(model):
            return indexes
        return score

    if score_name == "entropy_gt_index":
        scores = []
        for v in matrix_gt_index[:, originals_index]:
            count = Counter(v)
            count = [e / len(v) for e in count.values()]
            scores.append([- e * math.log(e, 2) for e in count if e != 0])
        indexes = sorted(
            list(range(len(matrix_gt_index))),
            key=lambda i: scores[i]
        )
        def score(model):
            return indexes
        return score
        
    if score_name == "diff_loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                f_base_indexes(model),
                key=lambda i: pure_model_infos[model][i][0] - pure_model_infos[model][i][1]
            )
            return indexes
        return score

    if score_name == "loss":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_loss.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                f_base_indexes(model),
                key=lambda i: pure_model_infos[model][i][0]
            )
            return indexes
        return score

    if score_name == "probability":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_probability.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                f_base_indexes(model),
                key=lambda i: pure_model_infos[model][i][0]
            )
            return indexes
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


#x = [int(e) for e in np.linspace(10, 50, 2)]

f_image_score = get_score(image_score_name)

results_to_save = {}

r = list(np.arange(0, 1, 0.05)) + [1]
x = []
for i in range(len(r) - 1):
    x.append((r[i], r[i + 1]))


matrix_gt_index = torch.Tensor(matrix_gt_index).long().transpose(1, 0)
matrix = torch.Tensor(matrix)


x_to_str = lambda e: "{}% --> {}%".format(e[0] * 100, e[1] * 100)
data_roc = {x_to_str(e): {"infos": [], "labels": [], "models": []} for e in x}
top_k_delegate = 0
for f in tqdm.tqdm(families):
    print("Family:", f)
    #print("Family:", families[f]["models"])
    # Reordered the images
    images_indexes = f_image_score(families[f]["pure"])

    alice_model = families[f]["pure"]
    #alice_model = get_delegate(families[f]["models"], families[f]["pure"])
    alice_model_ind = models.index(alice_model)


    matrix_gt_index_f = []
    if top_k_delegate > 0:
        for top_i in range(top_k_delegate):
            truth_f = matrix[:, alice_model_ind, top_i]

            matrix_gt_index_f_i = np.ones(matrix.shape[:2]) * top_k
            for row_i, row in enumerate(matrix):
                t = truth_f[row_i]
                models_ind, truth_ind = np.where(row == t)
                matrix_gt_index_f_i[row_i][models_ind] = truth_ind
            matrix_gt_index_f_i = torch.Tensor(matrix_gt_index_f_i.astype(int)).transpose(1, 0).long()
            matrix_gt_index_f.append(matrix_gt_index_f_i)


    for e in x:
        text = x_to_str(e)
        start = int(len(images_indexes) * e[0])
        end = int(len(images_indexes) * e[1])
        indexes = random.sample(images_indexes[start:end], args.n_images)
        #indexes = images_indexes[start:end][:args.n_images]
        v_o = [matrix_gt_index[alice_model_ind, indexes]]
        v_v = [matrix_gt_index[:, indexes]]
        for mat in matrix_gt_index_f:
            v_o.append(mat[alice_model_ind, indexes])
            v_v.append(mat[:, indexes])
        v_o = torch.cat(v_o, 0)
        v_o = v_o.unsqueeze(0).repeat_interleave(len(models), 0)

        v_v = torch.cat(v_v, 1)

        tmp = matrix[indexes, alice_model_ind, 0].unsqueeze(0)
        tmp = tmp.repeat_interleave(len(models), 0)
        o_eq_v = tmp == matrix[indexes, :, 0].transpose(1, 0)
        o_eq_v = o_eq_v.long()

        data_roc[text]["infos"] += list(model_distance(v_o, v_v, o_eq_v).cpu().numpy())
        data_roc[text]["models"] += list(models)
        for m in models:
            data_roc[text]["labels"].append(m in families[f]["models"])


score_fp_setted = {}
variants_roc = {}
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
    for l, inf in d:
        if not l:
            n_fp -= 1
        else:
            n_tp -= 1
        roc_fp.append(n_fp / n_neg)
        roc_tp.append(n_tp / n_positive)

        if n_fp / n_neg <= 0.01 and "0.01 - FP" not in score_fp_setted[e]:
            score_fp_setted[e]["0.01 - FP"] = n_fp / n_neg
            score_fp_setted[e]["0.01 - TP"] = n_tp / n_positive
        if n_fp / n_neg <= 0.05 and "0.05 - FP" not in score_fp_setted[e]:
            score_fp_setted[e]["0.05 - FP"] = n_fp / n_neg
            score_fp_setted[e]["0.05 - TP"] = n_tp / n_positive
        if n_fp / n_neg <= 0.1 and "0.1 - FP" not in score_fp_setted[e]:
            score_fp_setted[e]["0.1 - FP"] = n_fp / n_neg
            score_fp_setted[e]["0.1 - TP"] = n_tp / n_positive
    
    plt.plot(roc_fp, roc_tp, label=e)

    variants_roc[e] = {}
    variations = np.array([get_original_and_variation(m)[1] for m in data_roc[e]["models"]])
    for var in set(variations):
        var_indexes = (variations == var).nonzero()[0]

        labels = np.array(data_roc[e]["labels"])[var_indexes]
        scores = np.array(data_roc[e]["infos"])[var_indexes]
        d = sorted(list(zip(labels, scores)), key=lambda x: x[1])
        if distance_best == "min":
            d = sorted(d, key=lambda x: x[1], reverse=True)
        else:
            d = sorted(d, key=lambda x: x[1])

        n_tp = sum(labels)
        n_fp = len(d) - n_tp

        n_positive = n_tp
        n_neg = n_fp
        for l, _ in d:
            if not l:
                n_fp -= 1
            else:
                n_tp -= 1

            if n_fp / n_neg <= 0.05:
                variants_roc[e][var] = n_tp / n_positive
                break

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")

import pandas as pd
df = pd.DataFrame.from_dict(score_fp_setted) #, orient="column")
df.to_csv(os.path.join(output_dir, output_filename + ".csv"))


df = pd.DataFrame.from_dict(variants_roc) #, orient="column")
df.to_csv(os.path.join(output_dir, output_filename + "_variants.csv"))

"""
plt.bar(x, successes, width, color='b')
plt.bar(x, unknown, width, bottom=successes, color='grey')
plt.bar(x, fails, width, bottom=successes+unknown, color='r')
plt.xlabel("N Images")
plt.ylabel("Success Rate")
plt.legend(labels=['Success', 'Unknown', "Fail"])
plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")

np.save(os.path.join(output_dir, output_filename + ".npy"), results_to_save)
"""