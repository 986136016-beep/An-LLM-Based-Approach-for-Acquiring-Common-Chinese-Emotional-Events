
import requests
import re
import time
import os
import config
import openpyxl

BASE = r'./data'
OUT = r'.\ablation'
os.makedirs(OUT, exist_ok=True)

INDICATORS = ['遭受', '获得', '被授予', '打败', '免于', '被批评', '错吃', '丢失']

TEST_ONLY = 8

CLEAN_EXAMPLES = ['挫折', '好朋友的背叛', '陌生人的袭击', '校园暴力',
                  '无端的网络暴力', '社会的不公positive对待', '灭顶之灾', '失眠的困扰']

MODEL = config.MODELS["SparkDesk"]

def load_examples(indicator, n=8):

    examples = []
    for fname in [r'\自己举例generateevents用indicator1.xlsx', r'\自己举例generateevents用indicator2.xlsx']:
        wb = openpyxl.load_workbook(BASE + fname, read_only=True)
        ws = wb.active
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[0] and str(r[0]).strip() == indicator and r[1] is not None:
                examples.append(str(r[0]).strip() + str(r[1]).strip())
                if len(examples) >= n:
                    return examples

    if len(examples) < n:
        wb = openpyxl.load_workbook(BASE + r'\被+动标识符events.xlsx', read_only=True)
        ws = wb.active
        for r in ws.iter_rows(values_only=True):
            if r and r[0] and str(r[0]).strip().startswith(indicator):
                ev = str(r[0]).strip()
                if ev not in examples:
                    examples.append(ev)
                    if len(examples) >= n:
                        return examples
    return examples

def build_prompt_with_indicator(indicator, examples):
    lines = ["请输出 100 个能在日常生活neutral引发相应人类emotion的不同情感events。",
             "需确保generate短语的positive确性和多样性。以下yes一些例子："]
    for i, e in enumerate(examples, 1):
        lines.append("%d. %s" % (i, e))
    lines.append("")
    lines.append('请继续generate以"%s"开头的情感events，每个events占一行，不要编号。' % indicator)
    return "\n".join(lines)

def build_prompt_without_indicator(clean_examples):
    lines = ["请输出 100 个能在日常生活neutral引发相应人类emotion的不同情感events。",
             "需确保generate短语的positive确性和多样性。以下yes一些例子："]
    for i, e in enumerate(clean_examples, 1):
        lines.append("%d. %s" % (i, e))
    lines.append("")
    lines.append("请继续generate情感events，每个events占一行，不要编号。")
    return "\n".join(lines)

def call(prompt):
    for attempt in range(2):
        try:
            resp = requests.post(
                config.BASE_URL + "/chat/completions",
                headers={"Authorization": "Bearer " + config.API_KEY, "Content-Type": "application/json"},
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": config.DECODE["temperature"], "top_p": config.DECODE["top_p"],
                      "max_tokens": config.DECODE["max_tokens"], "stream": False},
                timeout=120,
            )
            if resp.status_code == 200:
                msg = resp.json()["choices"][0]["message"]
                content = msg.get("content")
                if not content:
                    if attempt == 0:
                        time.sleep(3); continue
                    return None
                return content
            return None
        except Exception:
            if attempt == 0:
                time.sleep(3); continue
            return None
    return None

def parse_events(text):
    events = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\s*\d+\s*[.、)）:：]\s*', '', line).strip()
        if line:
            events.append(line)
    return list(dict.fromkeys(events))[:100]

if __name__ == "__main__":
    results = {}
    for ind in INDICATORS[:TEST_ONLY]:
        examples = load_examples(ind)
        if len(examples) < 4:
            print("Warning: %s insufficient examples(%d)" % (ind, len(examples)))
            continue

        wout_path = os.path.join(OUT, "%s_without.txt" % ind)
        if os.path.exists(wout_path) and os.path.getsize(wout_path) > 500:

            events_without = [l.strip() for l in open(wout_path, encoding='utf-8') if l.strip()]
            n_without = len([e for e in events_without if e.startswith(ind)])

            text_with = call(build_prompt_with_indicator(ind, examples))
            n_with = len([e for e in parse_events(text_with or "") if e.startswith(ind)])
            results[ind] = (n_without, n_with)
            print("%-8s  w/o Indicator = %-4d   w/ Indicator = %d  (w/oreuse)" % (ind, n_without, n_with))
            continue

        p_with = build_prompt_with_indicator(ind, examples)
        text_with = call(p_with)
        n_with = len([e for e in parse_events(text_with or "") if e.startswith(ind)])

        p_without = build_prompt_without_indicator(CLEAN_EXAMPLES)
        text_without = call(p_without)
        events_without = parse_events(text_without or "")
        n_without = len([e for e in events_without if e.startswith(ind)])

        results[ind] = (n_without, n_with)
        print("%-8s  w/o Indicator = %-4d   w/ Indicator = %d" % (ind, n_without, n_with))

        with open(wout_path, "w", encoding="utf-8") as f:
            f.write("\n".join(events_without))

    print("\n===== Table 15 clean versionresult =====")
    print("%-10s %-16s %-16s" % ("indicator", "w/o Indicator", "w/ Indicator"))
    for ind in INDICATORS:
        if ind in results:
            n0, n1 = results[ind]
            print("%-10s %-16d %-16d" % (ind, n0, n1))
