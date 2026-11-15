
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score,
)

def find_best_threshold(probs, labels):

    precision, recall, thresholds = precision_recall_curve(labels, probs)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.argmax(f1))
    best_threshold = thresholds[idx]
    best_f1 = f1[idx]
    return best_threshold, best_f1, precision, recall, thresholds

def evaluate_at_threshold(probs, labels, threshold):

    pred = (probs >= threshold).astype(int)
    p = precision_score(labels, pred, zero_division=0)
    r = recall_score(labels, pred, zero_division=0)
    f = f1_score(labels, pred, zero_division=0)
    ap = average_precision_score(labels, probs)
    return p, r, f, ap

def run(val_probs, val_labels, test_probs, test_labels):

    best_thr, best_f1, prec, rec, thrs = find_best_threshold(val_probs, val_labels)
    print('=' * 60)
    print('validation set上选择threshold')
    print('=' * 60)
    print('最佳threshold(使 F1 max): %.4f' % best_thr)
    print('该threshold下的validation set F1: %.4f' % best_f1)

    p, r, f, ap = evaluate_at_threshold(test_probs, test_labels, best_thr)
    print()
    print('=' * 60)
    print('test set评估(threshold = %.4f)' % best_thr)
    print('=' * 60)
    print('PR-AUC = %.4f' % ap)
    print('Precision = %.4f' % p)
    print('Recall = %.4f' % r)
    print('F1 = %.4f' % f)

    return best_thr, (ap, p, r, f)

if __name__ == '__main__':

    import os
    val_probs = np.load('val_probs.npy')
    val_labels = np.load('val_labels.npy')
    test_probs = np.load('test_probs.npy')
    test_labels = np.load('test_labels.npy')

    run(val_probs, val_labels, test_probs, test_labels)
