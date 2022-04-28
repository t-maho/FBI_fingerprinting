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

from utils.model import get_model, get_original_and_variation
from utils.distances import get_model_distance

from sklearn.cluster import KMeans


###########################
###########################
### Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--info", type=int, default=3)
parser.add_argument("--max_drop", type=int, default=-0.15)
parser.add_argument("--n_delegate", type=int, default=2)
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
family = "pure"

remove_family_below = 1
print(model_distance_name)
if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/limited_information/identification_compound_pure_{}/{}".format(args.n_delegate, file_key)
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


def get_delegate(fmodels, n_delegate=1):
    print("uisqtnfdsiqgnfsuq")
    kmeans = KMeans(n_clusters=n_delegate)
    fm_i = [models.index(m) for m in fmodels]
    v_v = matrix_gt_index[fm_i]

    kmeans.fit(v_v)
    centers = kmeans.cluster_centers_

    centers = torch.Tensor(centers).unsqueeze(1).repeat_interleave(len(fmodels), 1).long()
    for c in centers:
        dists = model_distance(c, v_v)
        print(dists)
        c_i = dists.argmin()
        print(fmodels[c_i])





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
    #if args.n_delegate == 1:
    #    families[f]["delegate"] = families[f]["pure"]
    #else:
    families[f]["delegate"] = get_delegate(families[f]["models"], args.n_delegate)
    delegate_models.append(families[f]["delegate"])
    delegate_families.append(f)
delegate_models_indexes = [models.index(m) for m in delegate_models]

######################################
######################################
####### Select Compoung model

#possible_family = [f for f in families if args.variation in f ]
#family_compound = random.choice(possible_family)


def get_prediction_model(images, truth, param, family_compound):
    model_name = families[family_compound]["models"][0].split("-")
    model_name[-1] = param

    model = get_model("-".join(model_name))
    data_path = "/nfs/nas4/bbonnet/bbonnet/thibault/data/imagenet/test_224/"
    dataset = CustomTestImageDataset(data_path, order_from=images)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    for x, _ in data_loader:
        predictions.append(model(x.cuda(0) / 255).topk(top_k)[1].cpu())
    predictions = torch.cat(predictions, 0)

    predictions_gt = np.ones(len(images)) * top_k
    for i, p in enumerate(predictions):
        if truth[i] in p:
            predictions_gt = list(p).index(truth[i])

    return torch.Tensor(predictions_gt).long()

def get_range_param():
    if args.variation == "prune_last":
        range_ = (0, 0.3)
    if args.variation == "jpeg":
        range_ = (30, 100)
    else:
        raise ValueError
    return range_

######################
######################
## Distance between each model and delegate

#x = np.linspace(10, 300, 30)
#x = [int(e) for e in x]
x = [50, 100, 300]
results = {"successes": [], "fails": [], "unknown": []}

n_times = 1
n_compound = 20
highest_infos_labels_n_times = {n_images: [] for n_images in x}
for time_i in range(n_times):
    images_indexes = get_score(image_score_name)()
    highest_infos_labels = {n_images: [] for n_images in x}
    images_indexes_i = random.sample(images_indexes, max(x))
    for o in originals:
        delegate_families = []
        delegate_models = []
        for f in families:
            if o in families[f]["pure"]:
                delegate_models.append(families[f]["delegate"])
                delegate_families.append(f)
        delegate_models_indexes = [models.index(m) for m in delegate_models]
        mat_delegate = matrix_gt_index[delegate_models_indexes]

        for compound_i in tqdm.tqdm(range(n_compound)):
            param = get_range_param()
            r = random.uniform(param[0], param[1])

            m_v = get_prediction_model(names[images_indexes_i], truth[images_indexes_i], r, o)
            m_v = m_v.unsqueeze(0).repeat_interleave(len(delegate_models), 0)

            labels = [m in families[f]["models"] for f in delegate_families]

            for n_images in x:
                infos = model_distance(mat_delegate[:, images_indexes_i[:n_images]], m_v[:, images_indexes_i[:n_images]])
                if distance_best == "max":
                    best_i = infos.argmax()
                else:
                    best_i = infos.argmin()
                highest_infos_labels[n_images].append((float(infos[best_i]), delegate_families[best_i], family_compound))

    for n_images in x:
        highest_infos_labels_n_times[n_images].append(highest_infos_labels[n_images])


np.save(os.path.join(output_dir, output_filename + ".npy"), highest_infos_labels_n_times)
