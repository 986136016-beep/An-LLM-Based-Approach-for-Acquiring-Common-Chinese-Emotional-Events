# Common Chinese Emotional Events: Code and Data

This repository accompanies the paper **"An LLM-Based Approach for Acquiring Common Chinese Emotional Events"** (TALLIP). It provides the code, data, prompts, and documentation needed to reproduce the resource construction pipeline and the experiments reported in the paper.

---

## Overview

The paper constructs a large-scale resource of common Chinese emotional events. The final resource contains **102,218 common Chinese emotional events** with sentiment polarity labels, comprising:

- **44,282** events with non-neutral indicators
- **57,754** bei ("被") events
- **182** implicit emotional events

---

## Data Description

### Indicators

- `nonneutral_indicators.xlsx`: 726 non-neutral emotional event indicators (positive and negative), each with a polarity label.
- `bei_indicators.xlsx`: 918 compositional bei event indicators of the form "被+verb".

### Events

- `nonneutral_events.xlsx`: 44,282 events with non-neutral indicators, each with its emotion result.
- `bei_events.xlsx`: 57,754 bei events.
- `implicit_events.xlsx`: 182 implicit emotional events (events without indicators).

### Probe Set

- `probe_set.txt`: 3,097 emotion-cause clauses extracted from a Chinese emotion cause corpus [1], used for the coverage evaluation in Section 5.3.

---

## Prompts

The LLM prompt templates are provided in `data/prompt_templates.txt`. The core template is:

> Please output 100 different emotional events that can evoke corresponding human emotions in daily life. Ensure the correctness and diversity of the generated phrases. Below are some examples:
>
> 1. {example 1}
> ...
> 8. {example 8}
>
> Please continue generating emotional events starting with "[indicator]", one event per line, without numbering.

The full templates (including the bei-event and ablation variants) are documented in `data/prompt_templates.txt`.

---

## Model Versions and Inference Configuration

| Item | Value |
|---|---|
| Generation LLM | SparkDesk (spark-x2.6) |
| Compared LLMs | DeepSeek-V3 Flash, SparkDesk spark-x2.6, ERNIE Bot ernie-3.5, ChatGLM 4.7-Flash, Kimi k2.7, GPT-3.5-turbo, GPT-4o |
| Filter | RoBERTa-Large (hfl/chinese-roberta-wwm-ext-large) |
| temperature | 0.7 |
| top-p | 0.9 |
| max_tokens | 2048 |

