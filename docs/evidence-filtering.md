# Evidence Filtering

[Documentation](README.md) / Evidence Filtering

The evidence filtering model classifies whether a retrieved passage contains supporting evidence for a query.

| Item | Description |
| --- | --- |
| Model | [Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering) |
| Base model | Llama-3.1-8B-Instruct |
| Input | A query–passage pair |
| Output | `Yes` or `No` |
| Example | [`examples/evidence_filtering.py`](../examples/evidence_filtering.py) |

## Training data and evaluation

The [model card](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering#training-setup) describes training on **3,200 query–passage pairs** with expert-derived binary relevance labels and evaluation using **five-fold cross-validation**. The underlying data is available in [Medical RAG Expert Evaluation (200 queries)](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200), included alongside the model in the [Medical RAG collection](https://huggingface.co/collections/Yale-BIDS-Chen/medical-rag).

Use the `passages` subset's `relevant` field: `true` corresponds to `Yes`, and `false` to `No`. A passage is relevant if it fully or partially supports at least one must-have reference statement. Obtain question text from `responses` by joining on `query_id`, using one distinct question row per query so repeated model responses do not duplicate the training pairs.

For training settings and reported precision, recall, and F1, refer to the [current model card](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering#performance). See the [research guide](research/rethinking-med-rag.md#released-evaluation-data) for the evaluation tables and their row counts.

## Run the example

After [installation](installation.md), run from the repository root:

```bash
python examples/evidence_filtering.py
```

The example downloads the tokenizer and model weights, loads the model using bfloat16 with `device_map="auto"`, and classifies one predefined query–passage pair. It prints the generated label. Edit `query` and `doc` in the example to try another pair.

Use hardware with sufficient memory for an 8B model and its inference workload. If the model host requests authentication, follow the access instructions on the model page.

## Use with retrieval

Apply the classifier to each retrieved query–passage pair, then pass the retained evidence to the downstream experiment. The released example demonstrates the classifier separately; it is not automatically applied by `RetrievalSystem` or the generation code.

A `Yes` label is the model's evidence judgment, not an independent verification of a medical claim. Read the [project disclaimer](../README.md#disclaimer) before using model outputs.

For the study motivation and results, see the [research guide](research/rethinking-med-rag.md).
