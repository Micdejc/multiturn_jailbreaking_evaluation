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
- [Installation](#installation)
- [Human Annotation](#human-annotation)
- [LLM Evaluators](#llm-evaluators)
- [Embedding models](#embedding-models)
- [Experimental Results](#experimental-results)
- [Reproducibility](#reproducibility)
- [Citation](#Citation) 
- [License](#license)

---
## Installation

Please make sure to install the following python librairies as dependencies to run the program:

```bash
pip install jupyter pandas sentence-transformers xgboostpip
```

## Human Annotation
The ground truth for our experiments was created by a team of cybersecurity experts. They reviewed and provided judgments on a curated set of experimental conversations to ensure high-quality, reliable labels.  

The shared labeling guidelines and rules followed by the experts can be found [here](https://github.com/Micdejc/jailbreaking_assessment).  

---

## LLM Evaluators
Well-known LLM evaluators for safety evaluation were considered during the experiments, including the open-source [Llama Guard 3](https://huggingface.co/meta-llama/Llama-Guard-3-8B) and the closed-source GPT-4.1.

The targeted models were consistent with [*Grammatical Mirage Attack*](https://github.com/Micdejc/llm_multiturn_attacks/). Both open- and closed-source models were tested via Application Programming Interface (API) calls: closed-source models through paid subscriptions and open-source models via [LM Studio](https://lmstudio.ai/).

## Embedding models

The following sentence-embedding model were considered during the experimentation: [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [MPNet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and [RoBERTa-large-v1](https://huggingface.co/sentence-transformers/all-roberta-large-v1) 

---

## Experimental Results

Evaluations were conducted on **widely used adversarial benchmarks**:

- **[AdvBench](https://github.com/llm-attacks/llm-attacks)**
- **[HarmBench](https://github.com/centerforaisafety/HarmBench)**

## How to Run

You can use this framework to evaluate and moderate LLM responses for multi-turn jailbreaking in two ways:

1. **Automated Moderation with Existing LLM Evaluators**  
   Follow the instructions in [`run_moderator.ipynb`](run_moderator.ipynb) to moderate LLM responses using standard automated methods.

2. **Semantic Similarity-Based Evaluation (SIM)**  
   Follow the instructions in [`run_evaluator.ipynb`](run_evaluator.ipynb) to evaluate LLM responses based on semantic similarity, which provides human-aligned scoring.

📂 **Examples:**  
A few sample results from our experiments are available in the [`examples`](examples) folder for reference.

### Key Findings

- 🚀 Outperforms **open-source Llama-Guard 3** while achieving performance comparable to **closed-source GPT-4.1** as an evaluator
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
