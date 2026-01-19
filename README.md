# MultiCaption

## Overview

**MultiCaption** is a multilingual dataset designed for detecting disinformation through **contradictory visual claims**. The task focuses on determining whether two claims referring to the **same image or video** can both be true at the same time.

The dataset is motivated by real-world misinformation scenarios in which authentic images or videos are reused with misleading or false textual claims across different languages and platforms. Unlike prior benchmarks that are primarily English-only or text-centric, MultiCaption explicitly targets **multilingual and cross-lingual contradiction detection** grounded in visual context.

MultiCaption contains **11,088 labeled claim pairs across 64 languages**, making it the first large-scale dataset specifically constructed for this task.

---

## Dataset Information

- **Total claim pairs:** 11,088  
- **Languages covered:** 64  
- **Labels:**
  - **Contradicting (C):** Two claims cannot both be true for the same image or video.
  - **Non-Contradicting (NC):** Two claims are mutually consistent.

The dataset is available for research purposes only. Access can be requested at: https://doi.org/10.5281/zenodo.18230659

### Language Composition
- **Monolingual pairs:** 8,131  
- **Cross-lingual pairs:** 2,957  

### Train / Test Split

| Split | Contradicting | Non-Contradicting | Total Pairs | Languages |
|------|---------------|-------------------|-------------|-----------|
| Train | 4,020 | 4,767 | 8,795 | 59 |
| Test  | 2,415 | 2,505 | 4,920 | 52 |

The test set is **strictly disjoint** from the training set at the claim level to prevent relational leakage between splits.

---

## Key Features

- **Multilingual by design**  
  Claims originate from real-world fact-checking sources and social media, spanning 64 languages.

- **Visual-claim contradiction focus**  
  Claim pairs explicitly refer to the same image or video, targeting out-of-context and repurposed-media misinformation.

- **Multiple annotation strategies**  
  Labels are obtained using a combination of:
  - Manual expert validation  
  - Large Language Model (LLM) annotation  
  - Claim–post linking  
  - Graph-based self-expansion  

- **Realistic misinformation patterns**  
  Explicit negations and trivial refutations are filtered from the test set to better reflect real deployment scenarios.

---

## Model Variants

The paper reports benchmarks across three categories of models:

### Transformer-based Classifiers
- Multilingual BERT (mBERT)
- XLM-RoBERTa (XLM-R)
- Multilingual DeBERTa-v3 (mDeBERTa)

### Natural Language Inference (NLI) Models
- XLM-R-large-XNLI
- mDeBERTa-v3-mnli-xnli

### Large Language Models (LLMs)
- Phi-4-mini
- Mistral-7B-Instruct
- Llama-3.1-8B-Instruct
- Gemma-7B
- Qwen2.5-7B-Instruct  

LLMs are evaluated in both **zero-shot** and **fine-tuned** settings, with separate prompts for monolingual and multilingual configurations.

---

## Evaluation & Results

- **MultiCaption is more challenging than COSMOS**, a commonly used out-of-context detection benchmark.
- Zero-shot LLMs and fine-tuned NLI models struggle to achieve strong performance, indicating that the task goes beyond standard NLI semantics.
- **Fine-tuned transformer models outperform NLI models** on MultiCaption.
- **Fine-tuned LLMs achieve the strongest overall performance**, with Mistral showing the most consistent results across languages.

### Key Findings
- Task-specific fine-tuning is critical for strong performance.
- Multilingual training improves results for both transformers and LLMs.
- Fine-tuned LLMs exhibit stable performance across languages, while NLI-based models show higher variance.
- Text-only inputs are sufficient for detecting contradictory visual claims in most cases.

---

## Known Limitations

- **Text-only modeling**  
  Although claims are grounded in images or videos, the benchmark does not currently require visual feature inputs.

- **LLM-assisted annotations**  
  A subset of contradicting pairs is labeled using LLMs, which may introduce limited annotation noise despite high measured precision.

- **Class imbalance in raw data**  
  Naturally occurring contradicting pairs are less frequent and require data expansion strategies during training.

- **Contextual complexity**  
  Some contradictions depend on subtle temporal, cultural, or geopolitical context that remains challenging for current models.

## Citation

@misc{frade2026multicaptiondetectingdisinformationusing,
      title={MultiCaption: Detecting disinformation using multilingual visual claims}, 
      author={Rafael Martins Frade and Rrubaa Panchendrarajan and Arkaitz Zubiaga},
      year={2026},
      eprint={2601.11220},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={ https://arxiv.org/abs/2601.11220 }, 
}

## Contact information
For questions or suggestions, open an issue or contact:

**Rafael Martins Frade** – rafael.martins@newtral.es  
**Rrubaa Panchendrarajan** – r.panchendrarajan@qmul.ac.uk  
