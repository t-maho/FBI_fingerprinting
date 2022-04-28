import os
import numpy as np
import tqdm 
import torch
from collections import Counter
import random
import argparse

from utils.model import get_original_and_variation


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input predictions path")
parser.add_argument("--truth", required=True, help="Truth path")
parser.add_argument("--output_dir", type=float, default=-0.15)

parser.add_argument("--score", default="entropy", choices=["entropy", "worst-case", "mean-case"])
parser.add_argument("--family", default="pure", choices=["pure", "variation", "singleton"])
parser.add_argument("--max_drop", type=float, default=-0.15)
args = parser.parse_args()

n_times = 5
output_filename = "stop_{}-score_{}-max_drop_{}".format(args.family, args.score, args.max_drop)
os.makedirs(args.output_dir, exist_ok=True)


def score_mean_case(row, family_vec):
    score = []
    for label in set(row):
        label_i = (row == label).nonzero()[0] 
        p = len(label_i) / len(family_vec)
        score.append(len(set(family_vec[label_i])) * p)
    
    return np.sum(score)

def score_worst_case(row, family_vec):
    score = []
    for label in set(row):
        label_i = (row == label).nonzero()[0] #.flatten().cpu().numpy()
        score.append(len(set(family_vec[label_i])))
    return np.max(score)


fscore = eval("score_" + args.score.replace("-", "_"))
sorted_reversed = args.score in ["entropy"]



matrix = np.load(args.input, allow_pickle=True).item()
truth = np.array(np.load(args.truth))
models = list(matrix.keys())
matrix = np.array([matrix[m] for m in models])
if len(matrix.shape) == 2:
    matrix = np.expand_dims(matrix, -1)

accuracies = {}
for i, m in enumerate(models):
    accuracies[m] = (matrix[:, i, 0] == truth).sum()
    accuracies[m] /= len(truth)


################################
################################
##### Load DATA and truth

originals = set([get_original_and_variation(m)[0] for m in models])

indexes_kept = []
families = {}
families_models_vectors = []

for i, m in enumerate(models):
    o, m_v = get_original_and_variation(m)
    acc_drop = accuracies[m]  / accuracies[o] - 1
    if acc_drop > args.max_drop:
        if m_v is None:
            m_v = "No Variation"
        if args.family == "pure":
            f = o
        elif args.family == "variation":
            f = o + "/" + m_v
        elif args.family == "singleton":
            f = m
        else:
            raise ValueError("Unknown family: {}".format(args.family))

        if f not in families:
            families[f] = {"models": [], "indexes": []}
        families[f]["models"].append(m)
        families[f]["indexes"].append(len(indexes_kept))
        families_models_vectors.append(f)
        indexes_kept.append(i)


print("{} models kept".format(len(indexes_kept)))
models = models[indexes_kept]
matrix = matrix[:, indexes_kept]
print("Matrix Shape: {}".format(matrix.shape))


##########################
##########################
#### Glouton

results = []
remaining_indexes = list(range(len(models)))

matrix = np.array(matrix)
new_matrix = np.zeros(matrix.shape[:2])
for i in range(args.info):
    new_matrix += matrix[:, :, i] * 1000**i
matrix = new_matrix


results_n_times = []
for time_i in range(n_times):
    print("Time:", time_i)
    groups = [remaining_indexes]
    results = {m: {"images": [], "success": False} for m in models}
    while len(groups) > 0:
        print("\tLength Groups:", len(groups))
        new_groups = []
        for group in groups:
            #print("\t - Group Length:", len(group))
            scores = {}
            group_family = {}
            group_family_vec = []
            for g_i, g in enumerate(group):
                fg = families_models_vectors[g]
                if fg not in group_family:
                    group_family[fg] = []
                group_family[fg].append(g_i)
                group_family_vec.append(fg)


            for row_i, row in enumerate(matrix[:, group]):
                scores[row_i] = fscore(row, np.array(group_family_vec))

            top_score = sorted(scores.values(), key=lambda x: x)[0]
            scores_with_top_score = [k for k, v in scores.items() if v == top_score]
            random.shuffle(scores_with_top_score)
            best_row = scores_with_top_score[0]
            
            count = Counter(matrix[best_row][group])
            if len(count) == 1:
                print("WILL BE INFINTITE")
                remaining_models = [models[e] for e in group]
                print("Remaining Models {}:".format(remaining_models))
            else:
                for model_i in group:
                    results[models[model_i]]["images"].append(best_row)
                
                for k, v in count.items():
                    if v == 1:
                        i = np.where(matrix[best_row][group] == k)[0][0]
                        m = models[group[i]]
                        results[m]["success"] = True
                        #print("Success {} with {}".format(m, results[m]["images"]))
                    else:
                        indexes = np.where(matrix[best_row][group] == k)[0]
                        new_group = [group[i] for i in indexes]
                        new_group_family = [families_models_vectors[e] for e in new_group]

                        if len(set(new_group_family)) == 1:
                            for m in new_group:
                                results[models[m]]["success"] = True
                                #print("Success {} with {}".format(m, results[m]["images"]))
                        else:
                            new_groups.append(new_group)

        groups = new_groups
    
    results_n_times.append(
        [len(r["images"]) if r["success"] else "Fail" for r in results.values()]
    )
        
np.save(os.path.join(args.output_dir, output_filename + ".npy"), results_n_times)
