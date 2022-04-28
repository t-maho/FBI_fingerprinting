import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from collections import Counter
import argparse
import math
import time


###########################
###########################
### Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--info", type=int, default=3)
parser.add_argument(
    "--distance", type=str, default="mutual_information", 
    choices=["mutual_information", "l0", "l2", "mutual_distance"])
parser.add_argument(
    "--sort_images", type=str, default="random", 
    choices=["random", "probability", "diff_loss", "loss", "entropy_gt_index", "entropy_label"])
parser.add_argument("--n_estimation", type=int, default=1000)
args = parser.parse_args()

top_k = args.info
image_score_name = args.sort_images if args.sort_images.lower() not in ["none"] else None
model_distance_name = args.distance.lower()

if top_k == 1:
    file_key = "decision"
else:
    file_key = "top_{}".format(top_k)

output_dir = "/udd/tmaho/fingerprinting_real_images/very_limited_information/detection_estimation/{}".format(file_key)
#output_dir = "/srv/tempdd/tmaho/fingerprinting/very_limited_information/detection_estimation/{}".format(file_key)
os.makedirs(output_dir, exist_ok=True)
if image_score_name is not None:
    output_filename = "{}-images_sorted_{}".format(model_distance_name, image_score_name)
else:
    output_filename = "{}".format(model_distance_name)



###########################
###########################
### Score function to sort models



def score_mutual_information(o, v, to_distance=False):
    assert o.shape == v.shape
    mat = torch.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):    
            mat[i, j, i, j] = 1
            
    t = torch.cat([o.unsqueeze(2), v.unsqueeze(2)], dim=2)
    
    e = mat[tuple(t.reshape(-1, 2).transpose(1, 0))]    
    counts = e.reshape((t.shape[0], t.shape[1], 2, 2)).transpose(2, 3).sum(1)

    counts /= o.shape[1]
    p_o = counts.sum(1)
    p_v = counts.sum(2)

    h_o = - (p_o * torch.log2(p_o)).nan_to_num().sum(1)
    h_v = - (p_v * torch.log2(p_v)).nan_to_num().sum(1)
    

    h_o_v = - (counts * torch.log2(counts)).nan_to_num().sum([1, 2])
    mutual_information = h_o + h_v - h_o_v
    if to_distance:
        m, _ = torch.max(torch.cat((h_o.unsqueeze(1), h_v.unsqueeze(1)), dim=1), 1)
        mutual_information /= m
        return 1 - mutual_information.clip(0, 1)
    else:
        return mutual_information

def get_distance(distance_name):
    if distance_name == "l0":
        return  lambda x, y: (x != y).sum()
    elif distance_name == "l1":
        return  lambda x, y: (x - y).abs().sum()
    elif distance_name == "l2":
        return  lambda x, y: (x - y).norm()
    elif distance_name == "mutual_distance":
        return lambda x, y: score_mutual_information(x, y, to_distance=True)
    elif distance_name == "mutual_information":
        return lambda x, y: score_mutual_information(x, y, to_distance=False)
    else:
        raise ValueError("Unknown Distance:", distance_name)

model_distance = get_distance(model_distance_name)

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
            list(range(len(matrix_gt_index))),
            key=lambda i: scores[i]
        )
        s = int(len(indexes) * 0.8)
        def score(model):
            return indexes[s:]
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
                list(range(len(pure_model_infos[model]))),
                key=lambda i: pure_model_infos[model][i][0] - pure_model_infos[model][i][1]
            )
            s = int(len(indexes) * 0.1)
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
            e = int(len(indexes) * 0.4)
            return indexes[:e]
        return score

    if score_name == "probability":
        path = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/pure_model_probability.npy"
        pure_model_infos = np.load(path, allow_pickle=True).item()
        def score(model):
            indexes = sorted(
                list(range(len(pure_model_infos[model]))),
                key=lambda i: pure_model_infos[model][i][0]
            )
            s = int(len(indexes) * 0.1)
            e = int(len(indexes) * 0.4)
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


###########################
###########################
### Load Data

# Params
matrix = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/matrix_{}.npy".format(file_key)
matrix_decision = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/matrix_decision.npy"

models = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/models.npy"
names = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/fingerprinting/matrices/images.npy"
truth = "/nfs/nas4.irisa.fr/bbonnet/bbonnet/thibault/data/imagenet/test_ensemble_truth.npy"

models = np.load(models)
names = np.load(names)
matrix = np.load(matrix)
matrix_decision = np.load(matrix_decision)
truth = np.load(truth, allow_pickle=True).item()
print(len(models))
print("matrix shape", matrix.shape)
print("matrix_decision shape", matrix_decision.shape)

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

originals = []
originals_index = []
for i, e in enumerate(models):
    if not any(k in e for k in ["PRUNE", "JPEG", "HIST", "HALF", "POSTERIZE"]):
        originals.append(e)
        originals_index.append(i)


resorted_models = []
for o in originals:
    variants = []
    for m in models:
        if "-" + o + "-" in m or "HALF-" + o == m:
            variants.append(m)
    resorted_models += [o] + sorted(variants)



###########################
###########################
### Estimate threshold

def estimate_threshold(o, v, t=0.05):
    dists = []
    batch_size = 5000
    n_batch = math.ceil(args.n_estimation / batch_size)

    o = o.unsqueeze(0).repeat_interleave(batch_size, 0)
    dists = []
    for _ in range(n_batch):
        v_r = v.unsqueeze(0).repeat_interleave(batch_size, 0)
        for i in range(batch_size):
            v_r[i] = v_r[i][torch.randperm(len(v))]

        dists += list(model_distance(o.long(), v_r.long()).cpu().numpy())
    #print(np.array(dists))
    dists = sorted(dists)
    l = len(dists)
    #print("estimate_threshold", dists[-int(t * l) - 1])
    return dists[-int(t * l) - 1], dists


#####################################
#####################################



#x = [int(e) for e in np.linspace(10, 50, 2)]
f_image_score = get_score(image_score_name)


results_to_save = {}
models = list(models)

x = [20, 50, 100, 200]
images_indexes = {}
for o in originals:
    # Reordered the images
    indexes = f_image_score(o)
    images_indexes[o] = random.sample(indexes, max(x))

for n_images in x:
    #image_dir = "{}-images_sorted_{}-{}_images".format(model_distance_name, image_score_name, n_images)
    #image_dir = os.path.join(output_dir, image_dir)
    #os.makedirs(image_dir, exist_ok=True)

    print("{} images".format(n_images))
    results = {}
    for o in tqdm.tqdm(originals):
        data = {}
        results[o] = {}
        for m in resorted_models:
            data[m] = matrix_gt_index[images_indexes[o], models.index(m)]

        results[o]["estimation"] = estimate_threshold(data[o][:n_images], data[o][:n_images])[1]
        results[o]["max_estimation"] = max(results[o]["estimation"])
        results[o]["max_estimation_1%"] = sorted(results[o]["estimation"])[int(0.99 * args.n_estimation)]
        results[o]["max_estimation_5%"] = sorted(results[o]["estimation"])[int(0.95 * args.n_estimation)]

        #print("Estimation {}: {:.3f} / {:.3f} / {:.3f}".format(o, results[o]["max_estimation"], results[o]["max_estimation_1%"], results[o]["max_estimation_5%"]))
        v_o = data[o][:n_images].unsqueeze(0)
        v_o = v_o.repeat_interleave(len(resorted_models), 0)

        v_v = [data[m][:n_images].unsqueeze(0) for m in resorted_models]
        v_v = torch.cat(v_v, 0)
        results[o]["infos"] = list(model_distance(v_o.long(), v_v.long()).cpu().numpy())
        results[o]["label"] = ["-" + o + "-" in m or "HALF-" + o == m or m == o for m in resorted_models]
    
    infos_labels = []
    for o in results:
        for i in range(len(results[o]["infos"])):
            infos_labels.append((results[o]["infos"][i], results[o]["label"][i]))

    n_tp = sum([e for _, e in infos_labels])
    tp = n_tp
    fp = len(infos_labels) - n_tp
    infos_labels = sorted(infos_labels, key=lambda x: x[0])
    threshold = None
    for e_info, e_label in infos_labels:
        if e_label:
            tp -= 1
        else:
            fp -= 1
        if fp / (len(infos_labels) - n_tp) < 0.05:
            threshold = e_info
            break

    tp /= n_tp
    fp /= len(infos_labels) - n_tp
    print("TP: {} / FP: {}".format(tp, fp))
    print("Distance Threshold:", threshold)


    plt.clf()
    for k in ["max_estimation", "max_estimation_1%", "max_estimation_5%"]:
        tp = 0
        fp = 0
        for o in results:
            for i in range(len(results[o]["infos"])):
                if results[o]["infos"][i] > results[o][k] and results[o]["label"][i]:
                    tp += 1
                if results[o]["infos"][i] > results[o][k] and not results[o]["label"][i]:
                    fp += 1

        tp /= n_tp
        fp /= len(infos_labels) - n_tp
        print("{} --> TP: {} / FP: {}".format(k, tp, fp))

        d = [results[o][k] for o in results]
        plt.hist(d, alpha=0.3, label=k, bins=10, weights=np.ones(len(d)) / len(d))
    plt.axvline(x=threshold)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "estimation_hist_{}.pdf".format(n_images)), bbox_inches="tight")



"""
score_fp_setted = {}
for e in data_roc:
    d = list(zip(data_roc[e]["labels"], data_roc[e]["infos"]))
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
        if n_fp / n_neg <= 0.05:
            break
    print("n positive", n_tp / n_positive)
    plt.clf()
    plt.axvline(x=inf)
    plt.ylim(-0.1, 1.1)
    ests = data_roc[e]["estimation"]
    plt.hist(ests, bins=20, weights=np.ones(len(ests))/len(ests))
    plt.savefig(os.path.join(output_dir, "estimation_hist_{}.pdf".format(n_images)), bbox_inches="tight")

"""


#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.legend()
#plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")
#
#np.save(os.path.join(output_dir, output_filename + ".npy"), results_to_save)

