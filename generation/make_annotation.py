
import openpyxl
import os
import glob

OUT = r'.'
out_dir = OUT + r'\output'

records = []
for f in glob.glob(out_dir + r'\*.txt'):
    name = os.path.basename(f)[:-4]
    model, indicator = name.rsplit('_', 1)
    with open(f, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append((model, indicator, line))

print("总events数:", len(records))

wb_keep = openpyxl.Workbook()
ws = wb_keep.active
ws.append(['eventsID', 'events', '模型', 'indicator'])
for i, (m, ind, e) in enumerate(records, 1):
    ws.append([i, e, m, ind])
wb_keep.save(OUT + r'\D6_留档表.xlsx')

for stu in range(1, 4):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['eventsID', 'events', '有效(1)/无效(0)'])
    for i, (m, ind, e) in enumerate(records, 1):
        ws.append([i, e, ''])
    wb.save(OUT + r'\D6_标注表_学生%d.xlsx' % stu)

print("已generate: D6_留档表.xlsx + D6_标注表_学生1/2/3.xlsx")
