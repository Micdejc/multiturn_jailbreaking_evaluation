<p align="center" width="100%">
<img src="assets/CYSIM-JUDGE_illustration.png" alt="CYSIM-JUDGE" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<!-- [![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE) -->
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-31211/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# A Semantic Similarity-Based Approach for Human-Aligned Multi-Turn Jailbreaking Evaluation in Cybersecurity

## Overview

Reliable evaluation of multi-turn jailbreaking attacks is a critical yet unresolved challenge in Large Language Model (LLM) safety. Existing approaches frequently rely on **LLMs as automated judges** to reduce the cost of human evaluation. However, prior studies consistently reveal a **significant mismatch between LLM-based judgments and human assessments**, particularly when attackers exploit **linguistic nuances** such as verb tense manipulation.

This repository introduces an **open-source, semantic similarity–based evaluation framework** for multi-turn jailbreaking attacks, designed to deliver **trustworthy, human-correlated, and low–false-negative evaluation** suitable for **high-stakes cybersecurity contexts**.

### Core Idea

Instead of asking an LLM *“Is this a jailbreak?”*, this framework asks:

> **“How semantically close is the model’s response to a disallowed intent, as judged by humans?”**

We treat **human judgment as the ground truth signal** and design semantic similarity metrics that closely approximate it—across **multiple turns**, **implicit violations**, and **linguistic obfuscation**.

This study was conducted by [Michael Tchuindjang](https://github.com/Micdejc), [Nathan Duran](https://github.com/NathanDuran), [Phil Legg](https://github.com/pa-legg), and [Faiza Medjek](https://sciprofiles.com/profile/3778378) as part of a PhD research project in Cybersecurity and Artificial Intelligence, supported by a studentship at the University of the West of England (UWE Bristol), UK.

---

## Updates
- (XXXX-XX-XX) Insert any update here...
- (2026-01-08) Released the first version of the paper's dataset on GitHub.


## Table of Contents

- [LLM Evaluators](#llmevaluators)
- [Experimental Results](#experimentalresults)
- [Reproducibility](#reproducibility)
- [Citation](#Citation) 
- [License](#license)

---

## LLM Evaluators
Well-known LLM evaluators for safety evaluation were considered during the experiments, including the open-source [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) and the closed-source GPT-4.1.

The targeted models were consistent with [*Grammatical Mirage Attack*](https://github.com/Micdejc/llm_multiturn_attacks/). Both open- and closed-source models were tested via Application Programming Interface (API) calls: closed-source models through paid subscriptions and open-source models via [LM Studio](https://lmstudio.ai/).

---

## Experimental Results

Evaluations were conducted on **widely used adversarial benchmarks**:

- **[AdvBench](https://github.com/llm-attacks/llm-attacks)**
- **[HarmBench](https://github.com/centerforaisafety/HarmBench)**

### How to Run
- To moderate LLM responses for jailbreaking using automated methods, refer to the instructions provided in [`run_moderator.ipynb`](run_moderator.ipynb) file.

- To moderate LLM responses for jailbreaking using semantic similarities, refer to the instructions provided in [`run_evaluator.ipynb`](run_evaluator.ipynb) file.
  
A few examples of our experimental results are included in [`examples`](examples) folder.

**Notice:** If you would like to learn more about the implementation of the *Grammatical Mirage Attack*, feel free to [contact me](mailto:micdejc@gmail.com).

### Key Findings

- 🚀 Outperforms **open-source Llama-Guard 3** and even **closed-source GPT-4.1** as an evaluator
- 📉 Maintains **FNR ≤ 0.028 across all benchmarks**

<p align="center" width="100%">
<img src="assets/Confusion_matrices_2x2.png" alt="CONFUSION MATRIXES" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

These results demonstrate that **semantic similarity provides a more reliable and human-aligned evaluation signal than LLM judges**.

---
## Reproducibility

A note for hardware: all experiments we run use one or multiple NVIDIA GeForce RTX 4090 GPUs, which have 32GiB memory per chip. 

## Ethical & Security Notice

This repository is intended **strictly for defensive AI safety research**.  
It does **not** provide tools to generate, optimize, or deploy jailbreaking attacks.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{tchuindjang2026semanticjailbreakeval,
  title={Human-Correlated Semantic Evaluation of Multi-Turn Jailbreaking Attacks},
  author={Tchuindjang, Michael},
  year={2026},
  note={AI Safety and Cybersecurity Research}
}
```

## License
Copyright (c) 2025, Michael Tchuindjang 
All rights reserved.
