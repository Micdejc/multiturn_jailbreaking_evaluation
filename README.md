<p align="center" width="100%">
<img src="assets/CYSIM-JUDGE_illustration.png" alt="CYSIM-JUDGE" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

<!-- [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE) -->
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-31211/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# A Semantic Similarity-Based Approach for Human-Aligned Multi-Turn Jailbreaking Evaluation in Cybersecurity

## Overview

Reliable evaluation of multi-turn jailbreaking attacks is a critical yet unresolved challenge in Large Language Model (LLM) safety. Existing approaches frequently rely on **LLMs as automated judges** to reduce the cost of human evaluation. However, prior studies consistently reveal a **significant mismatch between LLM-based judgments and human assessments**, particularly when attackers exploit **linguistic nuances** such as verb tense manipulation.

This repository introduces an **open-source, semantic similarity–based evaluation framework** for multi-turn jailbreaking attacks, designed to deliver **trustworthy, human-correlated, and low–false-negative evaluation** suitable for **high-stakes cybersecurity contexts**.

This study was conducted by [Michael Tchuindjang](https://github.com/Micdejc), [Nathan Duran](https://github.com/NathanDuran), [Phil Legg](https://github.com/pa-legg), and [Faiza Medjek](https://sciprofiles.com/profile/3778378) as part of a PhD research project in Cybersecurity and Artificial Intelligence, supported by a studentship at the University of the West of England (UWE Bristol), UK.

---

## 🎯 Motivation

### Limitations of LLM-as-a-Judge Evaluation

While LLM-based evaluators are scalable, they suffer from critical weaknesses:

- ❌ Poor alignment with human judgment  
- ❌ Vulnerability to linguistic tricks (e.g., tense shifts, paraphrasing)  
- ❌ High false-negative rates in safety-critical scenarios  
- ❌ Reliance on closed-source, non-auditable models  

In cybersecurity, **missing a successful jailbreak is more dangerous than flagging a false positive**. Therefore, evaluation methods must prioritize **recall, robustness, and human alignment**.

---

## 🧠 Core Idea

Instead of asking an LLM *“Is this a jailbreak?”*, this framework asks:

> **“How semantically close is the model’s response to a disallowed intent, as judged by humans?”**

We treat **human judgment as the ground truth signal** and design semantic similarity metrics that closely approximate it—across **multiple turns**, **implicit violations**, and **linguistic obfuscation**.

---

## 🧪 Methodology

### 1. Multi-Turn Semantic Modeling

Each jailbreak attempt is represented as a sequence of turns:


Semantic signals are extracted **per assistant turn**, enabling detection of:
- Gradual safety erosion
- Partial violations
- Implicit intent realization

---

### 2. Reference Harmful Intent Space

A curated semantic intent space is constructed from:
- Policy-aligned disallowed behaviors
- Canonical harmful instructions
- Benchmark ground truth (AdvBench, HarmBench)

These references serve as **semantic anchors reflecting human-recognized harm**.

---

### 3. Semantic Similarity Scoring

For each assistant response, semantic similarity is computed against the harmful intent space using embedding-based metrics (e.g., cosine similarity).

This produces:
- Continuous risk scores (not binary labels)
- Robustness to paraphrasing and tense manipulation
- Sensitivity to semantic equivalence

---

### 4. Human-Correlated Evaluation

Automated scores are evaluated against **human judgments** using:
- Precision, recall, and F1 score
- False Negative Rate (FNR)
- False Positive Rate (FPR)
- FNR/FPR trade-off analysis

The objective is **maximal alignment with human evaluation**, not agreement with other LLMs.

---

## 📊 Experimental Results

Evaluations were conducted on **widely used adversarial benchmarks**:

- **AdvBench**
- **HarmBench**

### Key Findings

- 🚀 Outperforms **closed-source GPT-4.1** as an evaluator
- 📈 Improves true jailbreak detection by:
  - **+2.2% on AdvBench**
  - **+18.6% on HarmBench**
- 🧠 Best performance on **past-tense linguistic attacks**
- 🔍 Achieves:
  - **F1 = 0.67** on HarmBench (vs 0.65 for GPT-4.1)
  - **Recall = 0.993** on HarmBench (vs 0.935)
  - **F1 = 0.75** on AdvBench (vs 0.749)
  - **Recall = 0.992** on AdvBench (vs 0.864)
- 🛡️ Maintains **FNR ≤ 0.028 across all benchmarks**
- ⚖️ Significantly better **FNR/FPR trade-off** than GPT-4.1 (FNR up to 0.159)

These results demonstrate that **semantic similarity provides a more reliable and human-aligned evaluation signal than LLM judges**.

---

## ✅ Key Contributions

- Semantic similarity–based evaluation for multi-turn jailbreaking
- Strong correlation with human judgment
- State-of-the-art recall and low false-negative rates
- Robust to linguistic manipulation
- Fully open-source and auditable

---

## 🔬 Use Cases

- Evaluating multi-turn jailbreak datasets
- Benchmarking LLM safety and alignment
- Comparing defense mechanisms (e.g., RepE, fine-tuning)
- Reducing reliance on costly human evaluation
- Safety assessment in cybersecurity and high-risk domains

---

## 🛠️ Planned Extensions

- [ ] Human-in-the-loop calibration
- [ ] Explainable semantic attribution
- [ ] Turn-wise intent trajectory visualization
- [ ] Cross-model generalization analysis
- [ ] Public human-labeled evaluation subset

---

## ⚠️ Ethical & Security Notice

This repository is intended **strictly for defensive AI safety research**.  
It does **not** provide tools to generate, optimize, or deploy jailbreaking attacks.

---

## 📌 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{tchuindjang2026semanticjailbreakeval,
  title={Human-Correlated Semantic Evaluation of Multi-Turn Jailbreaking Attacks},
  author={Tchuindjang, Michael},
  year={2026},
  note={AI Safety and Cybersecurity Research}
}

