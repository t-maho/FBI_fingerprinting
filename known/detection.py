import os
import numpy as np
import random
import argparse

from utils.model import get_original_and_variation


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input predictions folder")
parser.add_argument("--output_dir", required=True, type=str)

parser.add_argument("--score", default="mean-case", choices=["best-case", "worst-case", "mean-case"])
parser.add_argument("--family", default="pure", choices=["pure", "variation", "singleton"])
parser.add_argument("--max_drop", type=float, default=-0.15)
args = parser.parse_args()

output_filename = "stop_{}-score_{}-max_drop_{}".format(args.family, args.score, args.max_drop)
os.makedirs(args.output_dir, exist_ok=True)


def score_mean_case(row, family_vec, model_interest):
    family_vec = np.array(family_vec) 
    model_indexes = (family_vec == model_interest).nonzero()[0]
    not_model_indexes = (family_vec != model_interest).nonzero()[0]
    labels_interest = set(row[model_indexes])
    score = []
    for label in labels_interest:
        not_f = (row[not_model_indexes] == label).sum()
        f = (row[model_indexes] == label).sum()
        score.append(not_f * f / len(model_indexes))

    return np.sum(score)

def score_worst_case(row, family_vec, model_interest):
    model_indexes = (family_vec == model_interest).nonzero()[0]
    not_model_indexes = (family_vec != model_interest).nonzero()[0]
    labels_interest = set(row[model_indexes])
    score = []
    for label in labels_interest:
        not_f = (row[not_model_indexes] == label).sum()
        f = (row[model_indexes] == label).sum()
        score.append(not_f * f / len(model_indexes))

    return np.max(score)


def score_best_case(row, family_vec, model_interest):
    family_vec = np.array(family_vec) 
    model_indexes = (family_vec == model_interest).nonzero()[0]
    not_model_indexes = (family_vec != model_interest).nonzero()[0]
    labels_interest = set(row[model_indexes])
    not_f_in = len(not_model_indexes)
    for label in labels_interest:
        not_f = (row[not_model_indexes] == label).sum()
        not_f_in -= not_f
    return -not_f_in
    


fscore = eval("score_" + args.score.replace("-", "_"))
sorted_reversed = args.score in ["entropy"]


matrix, models = [], []
truth = None
for filename in os.path.join(args.input):
    if filename == "ground_truth.npy":
        truth = np.array(np.load(os.path.join(args.input, filename)))

    matrix.append(np.load(os.path.join(args.input, filename)))
    models.append(filename[:-4])
matrix = np.array(matrix)
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
## Generate positive and negative couples

n_positives = 1000
families_names = list(families.keys())
couples = {f:[] for f in families_names}
for i in range(n_positives):
    family_name = random.choice(families_names)
    model = random.choice(families[family_name]["models"])
    couples[family_name].append((model, "positive"))

n_negatives = 1000
for i in range(n_negatives):
    family_name = random.choice(families_names)
    neg_family_name= random.choice([e for e in families_names if e != family_name])
    model = random.choice(families[neg_family_name]["models"])
    couples[family_name].append((model, "negative"))

##########################
##########################
### Matrix 3D to 2D
matrix = np.array(matrix)
new_matrix = np.zeros(matrix.shape[:2])
for i in range(args.info):
    new_matrix += matrix[:, :, i] * 1000**i
matrix = new_matrix

print("New Matrix shape:{}".format(matrix.shape))
models = list(models)
final_results = []
remaining_indexes = list(range(len(models)))
for f in couples:
    print("family: ", f)
    indexes_f = families[f]["indexes"]
    indexes_not_f = [i for i in list(range(len(models))) if i not in indexes_f]
    families_models_vectors_f = [e if e == f else "Singleton-{}".format(i) for i, e in enumerate(families_models_vectors)]
    families_models_vectors_f = np.array(families_models_vectors_f)

    init_scores = {}
    group_family = {}
    group_family_vec = []
    for f_i, fg in enumerate(families_models_vectors_f):
        if fg not in group_family:
            group_family[fg] = []
        group_family[fg].append(f_i)
        group_family_vec.append(fg)

    group_family_vec = np.array(group_family_vec)
    for row_i, row in enumerate(matrix):
        init_scores[row_i] = fscore(row, group_family_vec, f)

    for m_bb, expectation in couples[f]:
        index_m_bb = models.index(m_bb)
        pred_bb = matrix[:, index_m_bb]
        group = np.array(remaining_indexes)
        scores = init_scores
        images_submitted = []
        is_same = False
        while len(set(families_models_vectors_f[group])) > 1 and f in families_models_vectors_f[group] and not is_same: 
            if len(scores) == 0:
                group_family = {}
                group_family_vec = []
                for f_i, fg in enumerate(families_models_vectors_f[group]):
                    if fg not in group_family:
                        group_family[fg] = []
                    group_family[fg].append(f_i)
                    group_family_vec.append(fg)

                group_family_vec = np.array(group_family_vec)
                for row_i, row in enumerate(matrix[:, group]):
                    scores[row_i] = fscore(row, group_family_vec, f)

            top_score = sorted(scores.values(), key=lambda x: x)[0]
            scores_with_top_score = [k for k, v in scores.items() if v == top_score]
            image_index = scores_with_top_score[0]
            images_submitted.append(image_index)

            
            tmp_remaining_indexes = matrix[image_index, group] == pred_bb[image_index]
            tmp_remaining_indexes = tmp_remaining_indexes.nonzero()[0]
            group = group[tmp_remaining_indexes]

            # Check if the remaining models have exactly the same model
            is_same = True
            for i in range(len(group) - 1):
                if (matrix[:, group[i]] != matrix[:, group[i + 1]]).sum() > 1:
                    print("Not Same:", (matrix[:, group[i]] != matrix[:, group[i + 1]]).sum())
                    is_same = False
                    break
            scores = {}

        if f in families_models_vectors_f[group] and not any(e.startswith("Singleton") for e in families_models_vectors_f[group]):
            detection_results = "positive"
        elif f not in families_models_vectors_f[group] and len(group) > 0:
            detection_results = "negative"
        else:
            detection_results = "unknown"
    
        final_results.append({
            "family": f,
            "black_box": m_bb,
            "truth": expectation,
            "remaining_family_model": [models[e] for e in group if families_models_vectors_f[e] == f],
            "remaining_not_family_model": [models[e] for e in group if families_models_vectors_f[e] != f],
            "detection": detection_results,
            "images_submitted": images_submitted,
        })

np.save(os.path.join(args.output_dir, output_filename + ".npy"), final_results)
