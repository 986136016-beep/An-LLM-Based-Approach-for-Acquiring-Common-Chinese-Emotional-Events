
import numpy as np
import argparse
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

def bootstrap_ci(metric_fn, probs, labels, threshold, n_boot=1000, alpha=0.95, seed=42):

    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = metric_fn(probs[idx], labels[idx])
        scores.append(s)
    scores = np.array(scores)
    lo = np.percentile(scores, (1 - alpha) / 2 * 100)
    hi = np.percentile(scores, (1 + alpha) / 2 * 100)
    mean = scores.mean()
    return mean, lo, hi

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', type=str, default='D3')
    p.add_argument('--threshold', type=float, default=None,
                   help='分类threshold；不填则用validation set选出的最佳threshold(需从 result.txt 读)')
    args = p.parse_args()

    probs = np.load(args.prefix + '_test_probs.npy')
    labels = np.load(args.prefix + '_test_labels.npy')

    threshold = args.threshold
    if threshold is None:
        try:
            with open(args.prefix + '_result.txt', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('best_threshold(val):'):
                        threshold = float(line.split(':')[1].strip())
                        break
        except FileNotFoundError:
            pass
    if threshold is None:
        threshold = 0.5
    print('使用threshold: %.4f' % threshold)

    ap_mean, ap_lo, ap_hi = bootstrap_ci(
        lambda p, l: average_precision_score(l, p), probs, labels, threshold)

    pred = (probs >= threshold).astype(int)
    p_mean, p_lo, p_hi = bootstrap_ci(
        lambda p, l: precision_score(l, (p >= threshold).astype(int), zero_division=0),
        probs, labels, threshold)
    r_mean, r_lo, r_hi = bootstrap_ci(
        lambda p, l: recall_score(l, (p >= threshold).astype(int), zero_division=0),
        probs, labels, threshold)
    f_mean, f_lo, f_hi = bootstrap_ci(
        lambda p, l: f1_score(l, (p >= threshold).astype(int), zero_division=0),
        probs, labels, threshold)

    print()
    print('=' * 60)
    print('Bootstrap 95% confidence interval (1000 次有放回sampled)')
    print('=' * 60)
    print('PR-AUC   : %.4f  (95%% CI: [%.4f, %.4f])' % (ap_mean, ap_lo, ap_hi))
    print('Precision: %.4f  (95%% CI: [%.4f, %.4f])' % (p_mean, p_lo, p_hi))
    print('Recall   : %.4f  (95%% CI: [%.4f, %.4f])' % (r_mean, r_lo, r_hi))
    print('F1       : %.4f  (95%% CI: [%.4f, %.4f])' % (f_mean, f_lo, f_hi))

    with open(args.prefix + '_confidence interval.txt', 'w', encoding='utf-8') as f:
        f.write('threshold: %.4f\n' % threshold)
        f.write('PR-AUC: %.4f  [%.4f, %.4f]\n' % (ap_mean, ap_lo, ap_hi))
        f.write('Precision: %.4f  [%.4f, %.4f]\n' % (p_mean, p_lo, p_hi))
        f.write('Recall: %.4f  [%.4f, %.4f]\n' % (r_mean, r_lo, r_hi))
        f.write('F1: %.4f  [%.4f, %.4f]\n' % (f_mean, f_lo, f_hi))
    print('result已written %s_confidence interval.txt' % args.prefix)

if __name__ == '__main__':
    main()
