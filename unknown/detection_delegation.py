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

remove_family_below = 2

parser = argparse.ArgumentParser()
parser.add_argument("--info", type=int, default=3)
parser.add_argument("--max_drop", type=int, default=-0.15)
parser.add_argument("--delegate", default="far", choices=["far", "middle", "close"])
parser.add_argument("--n_images", default=20000, type=int)
parser.add_argument("--family", default="pure", choices=["pure", "variation", "singleton"])
parser.add_argument("--sort_images", type=str, default="random")
parser.add_argument("--gather_small", type=str, default="True")
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images
model_distance_name = args.distance.lower()
gather_small = args.gather_small.lower().strip() in ["true", "y"]

if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/detection_delegation/{}".format(file_key)
os.makedirs(output_dir, exist_ok=True)
output_filename = "stop_{}-{}-delegation_{}-gather_small_{}".format(args.family, model_distance_name, args.delegate, gather_small)

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
models = list(models)
names = np.load(names)
matrix = np.load(matrix)
truth = np.load(truth, allow_pickle=True).item()
print(len(models))
print("matrix shape", matrix.shape)

truth = np.array([truth[n] for n in names])

if top_k == 1:
    matrix = np.expand_dims(matrix, -1)

matrix_gt_index = np.ones(matrix.shape[:2]) * top_k
for row_i, row in tqdm.tqdm(enumerate(matrix)):
    t = truth[row_i]
    models_ind, truth_ind = np.where(row == t)
    matrix_gt_index[row_i][models_ind] = truth_ind
matrix_gt_index = matrix_gt_index.astype(int)
matrix_gt_index = torch.Tensor(matrix_gt_index)


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
matrix = matrix[:, models_indexes]
matrix_gt_index = matrix_gt_index[:, models_indexes]
models = models_kept
originals_index = [i for i, m in enumerate(models) if get_original_and_variation(m)[1] is None]

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
    if gather_small:
        for k in models_gathered:
            families["Single from {}".format(k)] = {"models": models_gathered[k], "pure": k}

    print("{} families.".format(len(families)))
print("Matrix Shape:", matrix.shape)

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


images_indexes = list(range(len(matrix)))
n_times = 20
matrix = torch.Tensor(matrix)
roc = {"TP": [], "FP": []}
for time_i in range(n_times):
    f_image_score = get_score(image_score_name)

    data_roc = {"infos": [], "labels": [], "models": []}
    for f in tqdm.tqdm(families):
        #print("Family:", f)
        #print("Family:", families[f]["models"])
        # Reordered the images
        alice_model = get_delegate(families[f]["models"], families[f]["pure"])
        alice_model_ind = models.index(alice_model)

        images_indexes = f_image_score(families[f]["pure"])
        if image_score_name.startswith("random_wrong_"):    
            indexes_right = list(np.nonzero(matrix[:, alice_model_ind, 0].numpy() == truth)[0])
            indexes_wrong = list(np.nonzero(matrix[:, alice_model_ind, 0].numpy() != truth)[0])

            random.shuffle(indexes_right)
            random.shuffle(indexes_wrong)

            n_wrong = int(args.n_images * float(image_score_name.split("_")[-1]))
            n_right = args.n_images - n_wrong
            images_indexes_n = indexes_wrong[:n_wrong] + indexes_right[:n_right]
        else:
            images_indexes_n = random.sample(images_indexes, args.n_images)

        #print("Alice Model: {}".format(models[alice_model_ind]))

        v_o = matrix_gt_index[alice_model_ind, images_indexes_n].unsqueeze(0)
        v_o = v_o.repeat_interleave(len(models), 0)
        v_v = matrix_gt_index[:, images_indexes_n]

        data_roc["infos"] += list(model_distance(v_o, v_v).cpu().numpy())
        data_roc["models"] += models

        for m in models:
            data_roc["labels"].append(m in families[f]["models"])

    print(Counter(data_roc["labels"]))

    d = list(zip(data_roc["labels"], data_roc["infos"]))
    if distance_best == "min":
        d = sorted(d, key=lambda x: x[1], reverse=True)
    else:
        d = sorted(d, key=lambda x: x[1])
    length = len(d)

    n_tp = sum(data_roc["labels"])
    n_fp = length - n_tp

    n_positive = n_tp
    n_neg = n_fp
    roc_fp, roc_tp = [], []
    for l, inf in d:
        if not l:
            n_fp -= 1
        else:
            n_tp -= 1
        roc_fp.append(n_fp / n_neg)
        roc_tp.append(n_tp / n_positive)

    roc["TP"].append(roc_tp)
    roc["FP"].append(roc_fp)


results_mean = {
    "TP_mean": np.mean(roc["TP"], 0),
    "TP_std": np.std(roc["TP"], 0),
    "FP_mean": np.mean(roc["FP"], 0),
    "FP_std": np.std(roc["FP"], 0)
}

for i, e in enumerate(results_mean["FP_mean"].flatten()):
    if e < 0.05:
        print("FP: {} / TP: {} (+/-{})".format(e,results_mean["TP_mean"][i], results_mean["TP_std"][i]))
        break

np.save(os.path.join(output_dir, output_filename + ".npy"), results_mean)

