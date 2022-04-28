import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

input_dir = "/srv/tempdd/tmaho/fingerprinting/very_limited_information/detection_estimation/decision"
output_dir = "/udd/tmaho/fingerprinting_real_images/very_limited_information/detection_estimation/decision"

estimation_ratio = 0.005

key_distance = "mutual_information"
key_image = "images_sorted_diff_loss-"
results = {}
for filename in os.listdir(input_dir):
    if key_distance not in filename or key_image not in filename:
        continue
    print("load", filename)
    n_images = filename.split("-")[-1]
    results[n_images] = np.load(os.path.join(input_dir, filename, "results.npy"), allow_pickle=True).item()


print("\n")
for e in sorted(list(results.keys())):

    all_infos = []
    all_labels = []
    for o in results[e]:
        all_infos += results[e][o]["infos"]
        all_labels += results[e][o]["label"]

    lists_thresholds_estimate= []
    plt.clf()
    print("N images:", e)
    length = len(all_labels)
    n_tp = sum(all_labels)
    n_fp = length - n_tp
    for estimation_ratio in tqdm.tqdm(np.linspace(0, 0.05, 20)):
        tp = 0
        fp = 0
        for o in results[e]:
            est_t = sorted(results[e][o]["estimation"])
            est_t = est_t[int(len(est_t) * (1 - estimation_ratio)) - 1]
            lists_thresholds_estimate.append(est_t)
            for i, l in enumerate(results[e][o]["label"]):
                if l and results[e][o]["infos"][i] > est_t:
                    tp += 1
                if results[e][o]["infos"][i] > est_t and not l:
                    fp += 1

        tp /= n_tp
        fp /= n_fp
        plt.plot(fp, tp, "x", "r")

    estimation_ratio=0.01
    tp = 0
    fp = 0
    for o in results[e]:
        est_t = sorted(results[e][o]["estimation"])
        est_t = est_t[int(len(est_t) * (1 - estimation_ratio)) - 1]
        for i, l in enumerate(results[e][o]["label"]):
            if l and results[e][o]["infos"][i] > est_t:
                tp += 1
            if results[e][o]["infos"][i] > est_t and not l:
                fp += 1

    tp /= n_tp
    fp /= n_fp
    print("Estimation --> TP: {} / FP: {}".format(tp, fp))
    
    fp_est = fp

    d = list(zip(all_labels, all_infos))
    d = sorted(d, key=lambda x: x[1])

    n_positive = n_tp
    n_neg = n_fp
    roc_fp, roc_tp = [], []
    score_05 = {}
    score_est = {}
    for l, inf in d:
        if not l:
            n_fp -= 1
        else:
            n_tp -= 1

        roc_fp.append(n_fp / n_neg)
        roc_tp.append(n_tp / n_positive)


        if n_fp / n_neg <= 0.05 and len(score_05) == 0:
            score_05["FP"] = n_fp / n_neg
            score_05["TP"] =  n_tp / n_positive
        if n_fp / n_neg <= fp_est and len(score_est) == 0:
            score_est["FP"] = n_fp / n_neg
            score_est["TP"] =  n_tp / n_positive


    plt.plot(roc_fp, roc_tp)
    plt.savefig(os.path.join(output_dir, "ROC_{}.pdf".format(e)), bbox_inches="tight")


    print("ROC 5% --> n positive", score_05)
    print("ROC Estimate--> n positive", score_est)
    plt.clf()
    plt.axvline(x=inf, ymin=-0.1, ymax=1.1)
    ests = lists_thresholds_estimate
    plt.hist(ests, bins=20, weights=np.ones(len(ests))/len(ests))
    plt.ylim(-0.1, 1.1)
    plt.savefig(os.path.join(output_dir, "estimation_hist_{}.pdf".format(e)), bbox_inches="tight")

    print("\n")
    print("\n")




#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.legend()
#plt.savefig(os.path.join(output_dir, output_filename + ".pdf"), bbox_inches="tight")
#
#np.save(os.path.join(output_dir, output_filename + ".npy"), results_to_save)


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