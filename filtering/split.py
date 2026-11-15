
import csv
import random
from collections import defaultdict

OUT = r'.'
SEED = 42
random.seed(SEED)

rows = []
with open(OUT + r'\D3_最终train数据.csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append((int(r['label']), r['event'], r['indicator']))

print('total据:', len(rows))

pos = [(l, e, i) for l, e, i in rows if l == 1]
neg = [(l, e, i) for l, e, i in rows if l == 0]
print('positive样本:', len(pos), ' negative样本:', len(neg))

by_ind = defaultdict(list)
for label, event, ind in pos:
    by_ind[ind].append((label, event, ind))

inds = list(by_ind.keys())
random.shuffle(inds)
n = len(inds)
n_train = int(n * 0.80)
n_val = int(n * 0.10)

train_inds = set(inds[:n_train])
val_inds = set(inds[n_train:n_train + n_val])
test_inds = set(inds[n_train + n_val:])
print('positive样本indicatorsplit: train=%d val=%d test=%d' % (len(train_inds), len(val_inds), len(test_inds)))

train_pos = []
val_pos = []
test_pos = []
for ind in train_inds:
    train_pos.extend(by_ind[ind])
for ind in val_inds:
    val_pos.extend(by_ind[ind])
for ind in test_inds:
    test_pos.extend(by_ind[ind])

random.shuffle(neg)
n_neg = len(neg)
n_neg_train = int(n_neg * 0.80)
n_neg_val = int(n_neg * 0.10)
train_neg = neg[:n_neg_train]
val_neg = neg[n_neg_train:n_neg_train + n_neg_val]
test_neg = neg[n_neg_train + n_neg_val:]

print('negative样本split: train=%d val=%d test=%d' % (len(train_neg), len(val_neg), len(test_neg)))

train = train_pos + train_neg
val = val_pos + val_neg
test = test_pos + test_neg
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print('最终split: train=%d val=%d test=%d' % (len(train), len(val), len(test)))

for name, data in [('train', train), ('val', val), ('test', test)]:
    p = sum(1 for r in data if r[0] == 1)
    q = sum(1 for r in data if r[0] == 0)
    print('  %s: positive=%d negative=%d' % (name, p, q))

def write_csv(path, data):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'event', 'indicator'])
        for r in data:
            w.writerow(r)

write_csv(OUT + r'\D3_train.csv', train)
write_csv(OUT + r'\D3_val.csv', val)
write_csv(OUT + r'\D3_test.csv', test)

inter_tv = train_inds & val_inds
inter_tt = train_inds & test_inds
inter_vt = val_inds & test_inds
print()
print('positive样本indicator交集校验: train∩val=%d train∩test=%d val∩test=%d (应全为0)' % (
    len(inter_tv), len(inter_tt), len(inter_vt)))

print('已done第2步: D3_train.csv / D3_val.csv / D3_test.csv')
