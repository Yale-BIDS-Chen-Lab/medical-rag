# Rethinking Retrieval-Augmented Generation for Medicine

[Research](README.md) / Rethinking Medical RAG

**Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights**

[Paper](https://arxiv.org/abs/2511.06738) · [Dataset](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200) · [Evidence filtering model](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering) · [Hugging Face collection](https://huggingface.co/collections/Yale-BIDS-Chen/medical-rag)

## Study overview

The study evaluates medical RAG through three stages: evidence retrieval, evidence selection, and response generation. The released dataset contains 200 questions: 100 patient questions from [K-QA](https://github.com/Itaymanes/K-QA) and 100 USMLE Step 2/3-style questions from [MedBullets](https://github.com/HanjieChen/ChallengeClinicalQA). It includes 800 responses from GPT-4o and Llama-3.1-8B, each with and without RAG, and the top-16 MedCPT passages for each question.

The evaluation identifies retrieval and evidence selection as key failure points. The paper then studies evidence filtering and query reformulation as practical ways to improve medical RAG. See the paper for the experimental settings, comparisons, and results.

## Released evaluation data

The [dataset card](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200#data-organization) documents six tables, all available through the `validation` split:

| Subset | One row represents | Rows |
| --- | --- | ---: |
| `responses` | One question–model–condition response | 800 |
| `passages` | One retrieved passage for one question | 3,200 |
| `retrieval` | One must-have reference statement paired with a passage | 30,784 |
| `selection` | One model-generated reference paired with a candidate passage | 26,032 |
| `factuality` | One evaluated statement from a model response | 15,970 |
| `completeness` | One response paired with a must-have reference statement | 7,696 |

The four evaluation tables contain **80,482 rows**. Including `responses` and `passages`, the release contains **84,482 rows** in total. These are table row counts; responses and retrieved passages are source records. The reference answers contain 1,924 must-have statements used in retrieval and completeness evaluation.

These counts reflect the Hugging Face release checked on October 28, 2026. Use its [dataset card](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200) and [included evaluator](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200#loading-and-scoring) for release-specific schemas, scoring definitions, and expected results.

```python
from datasets import load_dataset

responses = load_dataset(
    "Yale-BIDS-Chen/medical-rag-expert-evaluation-200",
    "responses",
    split="validation",
)
```

This is evaluation data with retrieved candidate passages and labels. The [retrieval examples](../retrieval.md) use separately prepared search corpora and indexes. For the data's source attribution and reuse terms, refer to the [dataset card](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200#license).

## Code map

| Component | Code | Guide |
| --- | --- | --- |
| Single-query retrieval | [`examples/quickstart.py`](../../examples/quickstart.py) | [Retrieval](../retrieval.md) |
| MedQA retrieval | [`examples/retrieval.py`](../../examples/retrieval.py) | [Retrieval](../retrieval.md) |
| Evidence filtering | [`examples/evidence_filtering.py`](../../examples/evidence_filtering.py) | [Evidence filtering](../evidence-filtering.md) |
| Rationale querying | [`examples/retrieval_using_rationale_query.py`](../../examples/retrieval_using_rationale_query.py) | [Query reformulation](../query-reformulation.md) |
| Corpus preparation and indexing | [`src/data/`](../../src/data), [`src/utils.py`](../../src/utils.py) | [Data and indexing](../data-and-indexing.md) |
| Generation and prompts | [`src/medrag.py`](../../src/medrag.py), [`src/template.py`](../../src/template.py) | Inspect source for model-specific configuration |

## Implementation scope

The repository provides component examples for experimentation. The examples use a single query, the MedQA test split, or a predefined query–passage pair; they do not reproduce the full expert annotation study or all reported benchmark results with one command. The released evaluation dataset includes its own evaluator, documented on its Hugging Face page.

Evidence filtering and rationale querying are demonstrated separately. Connecting them to retrieval and generation requires explicit integration in the experiment. This distinction also applies to the conceptual overview in the main README.

For a study-specific experiment, record the model and corpus revisions, retrieval settings, prompts, and evaluation procedure together with the results.

## Citation

The BibTeX entry is available in the [main README](../../README.md#citation).
