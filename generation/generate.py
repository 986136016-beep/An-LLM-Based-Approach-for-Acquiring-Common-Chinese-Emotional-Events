
import requests
import json
import re
import config

def load_examples(indicator, path, n=8):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    examples = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] and str(r[0]).strip() == indicator and r[1] is not None:
            examples.append(str(r[0]).strip() + str(r[1]).strip())
            if len(examples) >= n:
                break
    return examples

def build_prompt(indicator, examples):

    lines = [
        "请输出 100 个能在日常生活neutral引发相应人类emotion的不同情感events。",
        "需确保generate短语的positive确性和多样性。以下yes一些例子：",
    ]
    for i, e in enumerate(examples, 1):
        lines.append("%d. %s" % (i, e))
    lines.append("")
    lines.append('请继续generate以"%s"开头的情感events，每个events占一行，不要编号，直接给出events本身。' % indicator)
    return "\n".join(lines)

def call_llm(model, prompt):
    resp = requests.post(
        config.BASE_URL + "/chat/completions",
        headers={
            "Authorization": "Bearer " + config.API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.DECODE["temperature"],
            "top_p": config.DECODE["top_p"],
            "max_tokens": config.DECODE["max_tokens"],
            "stream": False,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        print("error:", resp.status_code, resp.text[:300])
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def parse_events(text):

    events = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        line = re.sub(r'^\s*\d+\s*[.、)）:：]\s*', '', line)
        line = line.strip()
        if line:
            events.append(line)
    return events

if __name__ == "__main__":
    INDICATOR = "按错"
    EX_PATH = r'./data\自己举例generateevents用indicator1.xlsx'

    examples = load_examples(INDICATOR, EX_PATH)
    print("few-shot 示例(%d个):" % len(examples))
    for e in examples:
        print("  ", e)

    prompt = build_prompt(INDICATOR, examples)
    print("\n===== prompt =====")
    print(prompt)
    print("==================\n")

    for name, model in config.MODELS.items():
        print(">>> 调用 %s (%s)" % (name, model))
        text = call_llm(model, prompt)
        if text is None:
            continue
        events = parse_events(text)

        events = list(dict.fromkeys(events))
        print("generate %d 个(deduplicated后)" % len(events))
        for e in events[:10]:
            print("  ", e)

        with open(r".\_test_%s.txt" % name, "w", encoding="utf-8") as f:
            f.write("\n".join(events))
        print("已save到 _test_%s.txt\n" % name)
