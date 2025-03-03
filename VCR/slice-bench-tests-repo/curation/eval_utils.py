import numpy as np
from sklearn.metrics import precision_score
from scipy.stats import rankdata

# Credit to: https://github.com/HazyResearch/domino/blob/5284ad826f9d98cd8cb3e367bac26f66c390c687/domino/eval/metrics.py#L55
def precision_at_k_single_slice(slice_gt, slicer_prob, k):
    '''
    Compute precision at k for each slice in slicer_prob. but only for the slice specified by slice_gt.
    '''
    precisions = []

    slice_gt = slice_gt.astype(bool)
    for i in range(slicer_prob.shape[1]):
        probs = slicer_prob[:, i]
        probs = rankdata(-probs, method="ordinal") <= k
        precisions.append(precision_score(slice_gt, probs))

    return precisions
