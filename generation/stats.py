
import openpyxl
from collections import Counter, defaultdict

OUT = r'.'

wb = openpyxl.load_workbook(OUT + r'\D6_留档表.xlsx')
ws = wb.active
meta = {}
for r in ws.iter_rows(min_row=2, values_only=True):
    if r[0] is not None:
        meta[int(r[0])] = (r[2], r[3])

students = {}
for s in range(1, 4):
    wb = openpyxl.load_workbook(OUT + r'\D6_标注表_学生%d.xlsx' % s)
    ws = wb.active
    d = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is not None and r[2] is not None and str(r[2]).strip() != '':
            d[int(r[0])] = int(r[2])
    students[s] = d

ids = sorted(meta.keys())
vote_by_model = defaultdict(lambda: Counter())

for eid in ids:
    model, ind = meta[eid]
    votes = [students[s].get(eid) for s in range(1, 4)]
    votes = [v for v in votes if v is not None]
    if not votes:
        continue
    maj = Counter(votes).most_common(1)[0][0]
    vote_by_model[model][maj] += 1

lines = []
lines.append('=' * 60)
lines.append('D6 各模型有效率对比 (3人majority vote)')
lines.append('=' * 60)
lines.append('')

for model in vote_by_model:
    c = vote_by_model[model]
    valid = c.get(1, 0)
    invalid = c.get(0, 0)
    total = valid + invalid
    acc = valid / total if total else 0
    lines.append('【%s】' % model)
    lines.append('  有效 = %d   无效 = %d   有效率 = %.1f%%   幻觉率 = %.1f%%' % (
        valid, invalid, 100*acc, 100*(1-acc)))
    lines.append('')

lines.append('=' * 60)
lines.append('conclusion: 有效率最高(幻觉率最低)的模型为优选。')

txt = '\n'.join(lines)
open(OUT + r'\D6_模型对比result.txt', 'w', encoding='utf-8').write(txt)
print(txt)
