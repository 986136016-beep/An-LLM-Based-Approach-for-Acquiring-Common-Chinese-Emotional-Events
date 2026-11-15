
import openpyxl
from collections import Counter

OUT = r'.'

POL_MAP = {'positive': 'positive', 'positive': 'positive', '1': 'positive', 'pos': 'positive', 'positive': 'positive',
           'negative': 'negative', 'negative': 'negative', '0': 'negative', 'neg': 'negative', 'negative': 'negative',
           'neutral': 'neutral', 'neutral': 'neutral', '2': 'neutral', '无': 'neutral', 'none': 'neutral'}

def norm(v):
    if v is None:
        return 'neutral'
    return POL_MAP.get(str(v).strip(), 'neutral')

wb = openpyxl.load_workbook(OUT + r'\D2a_indicator作者对照表.xlsx')
ws = wb.active
author = {int(r[0]): {'ind': r[1], 'pol': norm(r[2])}
          for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None}
ids = sorted(author.keys())

student = {}
for s in range(1, 4):
    wb = openpyxl.load_workbook(OUT + r'\D2a_indicator标注表_学生%d.xlsx' % s)
    ws = wb.active
    student[s] = {int(r[0]): norm(r[2]) for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None}

CATS = ['positive', 'negative', 'neutral']

def majority(votes):
    c = Counter(votes)
    m = c.most_common(2)
    if len(m) == 1 or m[0][1] > m[1][1]:
        return m[0][0]
    return 'disagreement'

def fleiss(counts, n_raters=3, n_cat=3):
    N = len(counts)
    if N == 0:
        return 0.0
    Pi = [(sum(c*c for c in cnt) - n_raters)/(n_raters*(n_raters-1)) for cnt in counts]
    P_bar = sum(Pi)/N
    Pe = sum((sum(cnt[j] for cnt in counts)/(N*n_raters))**2 for j in range(n_cat))
    return 1.0 if (1-Pe)==0 else (P_bar-Pe)/(1-Pe)

correct = Counter(); total = Counter(); disagree = []
for i in ids:
    a = author[i]
    votes = [student[s].get(i, 'neutral') for s in range(1, 4)]
    maj = majority(votes)
    total[a['pol']] += 1
    if maj == a['pol']:
        correct[a['pol']] += 1
    else:
        disagree.append((i, a['ind'], a['pol'], maj, votes))

lines = []
lines.append('=' * 60)
lines.append('D2a indicator层独立标注auditresult')
lines.append('=' * 60)
lines.append('')
lines.append('【学生majority vote vs 作者标注 agreement rate】')
for c in CATS:
    if total[c]:
        lines.append('  %s: %d/%d = %.2f%%' % (c, correct[c], total[c], 100*correct[c]/total[c]))
acc = sum(correct.values())/sum(total.values()) if sum(total.values()) else 0
lines.append('  总体: %d/%d = %.2f%%' % (sum(correct.values()), sum(total.values()), 100*acc))
lines.append('')

cat_counts = []
for i in ids:
    votes = [student[s].get(i, 'neutral') for s in range(1, 4)]
    cat_counts.append([votes.count(c) for c in CATS])
kappa = fleiss(cat_counts)
lines.append('【三人标注agreement Fleiss Kappa】= %.4f' % kappa)
lines.append('')
lines.append('【disagreementindicator清单】 共 %d 个' % len(disagree))
for (i, ind, ap, mp, votes) in disagree:
    lines.append('  [%d] %s | 作者=%s 学生多数=%s | 三人=%s' % (i, ind, ap, mp, votes))

txt = '\n'.join(lines)
open(OUT + r'\D2a_indicatorauditresult.txt', 'w', encoding='utf-8').write(txt)
print(txt)
