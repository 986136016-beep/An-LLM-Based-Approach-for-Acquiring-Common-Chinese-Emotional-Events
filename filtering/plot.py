
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def load(prefix):
    probs = np.load(prefix + '_val_probs.npy')
    labels = np.load(prefix + '_val_labels.npy')
    return probs, labels

nn_probs, nn_labels = load('D3_nonneutral')
bei_probs, bei_labels = load('D3_bei')

nn_prec, nn_rec, _ = precision_recall_curve(nn_labels, nn_probs)
bei_prec, bei_rec, _ = precision_recall_curve(bei_labels, bei_probs)

nn_ap = average_precision_score(nn_labels, nn_probs)
bei_ap = average_precision_score(bei_labels, bei_probs)

plt.figure(figsize=(7, 6))
plt.plot(nn_rec, nn_prec, 'b-', linewidth=2, label='non-neutral indicators (AP=%.3f)' % nn_ap)
plt.plot(bei_rec, bei_prec, 'orange', linewidth=2, label='bei events (AP=%.3f)' % bei_ap)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curves of the filter (Figure 5)')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('D3_PR曲线.png', dpi=150)
print('已save D3_PR曲线.png')
print('非neutral AP=%.4f, bei AP=%.4f' % (nn_ap, bei_ap))
