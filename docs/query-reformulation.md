# Query Reformulation

[Documentation](README.md) / Query Reformulation

Rationale querying uses a model-generated explanation of a question as the retrieval query. This provides a way to study how richer query formulations affect the retrieved evidence.

## Workflow

1. Load the original questions.
2. Generate rationale-style responses with an instruction-tuned model.
3. Use those generated responses as queries to the retriever.
4. Inspect or use the returned passages in a downstream experiment.

## Run the example

After [installation](installation.md), run from the repository root:

```bash
python examples/retrieval_using_rationale_query.py
```

The example uses:

| Setting | Default |
| --- | --- |
| Questions | MedQA test split |
| Reformulation model | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Generation engine | vLLM |
| Retriever | MedCPT |
| Corpus | Textbooks |
| Retrieved passages | 16 per query |

This runs over the full question split. It generates rationale-style queries first and then performs retrieval, printing the original question, generated query, and top passage for the first question. It does not compute answer accuracy.

The script uses one GPU (`tensor_parallel_size=1`) and bfloat16. Model access, available GPU memory, and the configured context length must suit the selected model. Update the model path and vLLM settings inside the example when adapting it.

Generated rationales may contain errors; treat them as experimental retrieval inputs. The example does not automatically apply evidence filtering.

## References

- [Rethinking Retrieval-Augmented Generation for Medicine](https://arxiv.org/abs/2511.06738)
- [Rationale-Guided Retrieval Augmented Generation for Medical Question Answering](https://aclanthology.org/2025.naacl-long.635/)

[Retrieval](retrieval.md) · [Research guide](research/rethinking-med-rag.md)
