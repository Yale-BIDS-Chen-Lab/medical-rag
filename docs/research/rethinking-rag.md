# Rethinking Retrieval-Augmented Generation for Medicine

[Documentation](../README.md) / Research

**Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights**  
Hyunjae Kim, Jiwoong Sohn, Aidan Gilson, et al. · 2025

[Paper](https://arxiv.org/abs/2511.06738) · [Evidence filtering model](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering)

## Study overview

The study evaluates medical RAG through three stages: evidence retrieval, evidence selection, and response generation. Eighteen medical experts contributed 80,502 annotations across 800 model outputs for 200 real-world patient and USMLE-style questions.

The evaluation identifies retrieval and evidence selection as key failure points. The paper then studies evidence filtering and query reformulation as practical ways to improve medical RAG. See the paper for the experimental settings, comparisons, and results.

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

The repository provides component examples for experimentation. The examples use the MedQA test split or a predefined query–passage pair; they do not reproduce the full expert annotation study or all reported benchmark results with one command.

Evidence filtering and rationale querying are demonstrated separately. Connecting them to retrieval and generation requires explicit integration in the experiment. This distinction also applies to the conceptual overview in the main README.

For a study-specific experiment, record the model and corpus revisions, retrieval settings, prompts, and evaluation procedure together with the results.

## Citation

The BibTeX entry is available in the [main README](../../README.md#citation).
