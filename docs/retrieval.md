# Retrieval

[Documentation](README.md) / Retrieval

Retrieve passages from a medical corpus with sparse or dense retrievers, or combine dense and sparse retrieval with reciprocal rank fusion (RRF).

## One query

After [installation](installation.md), run:

```bash
python examples/quickstart.py \
  --query "What is the role of insulin in glucose metabolism?" \
  --retriever MedCPT --corpus Textbooks --k 3
```

The example prints each passage's title, ID, retrieval score, and content. It searches one question and does not run a question-answering benchmark. Corpus and index files are stored under `./corpus`; change the location with `--db-dir`.

## Python usage

```python
from src.utils import RetrievalSystem

retriever = RetrievalSystem(
    retriever_name="MedCPT",
    corpus_name="Textbooks",
    db_dir="./corpus",
    cache=False,
)
passages, scores = retriever.retrieve(
    question="What is the role of insulin in glucose metabolism?",
    k=3,
)
```

`passages` is a list of dictionaries containing passage fields such as `id`, `title`, and `content`. `scores` contains the corresponding retrieval scores. Scores depend on the retriever and should not be compared directly across different retrieval methods.

Setting `cache=True` also builds or loads a passage-text cache in memory; use this when repeated retrieval justifies the additional memory. Indexes are saved to disk with either setting.

## Retriever names

Use these exact names with `--retriever` or `retriever_name`:

| Name | Implementation |
| --- | --- |
| `BM25` | Sparse retrieval through Pyserini |
| `Contriever` | `facebook/contriever` |
| `SPECTER` | `allenai/specter` |
| `MedCPT` | `ncbi/MedCPT-Query-Encoder`, paired with its article encoder |
| `Qwen3-0.6B` | `Qwen/Qwen3-Embedding-0.6B` |
| `RRF-2` | BM25 + MedCPT |
| `RRF-4` | BM25 + Contriever + SPECTER + MedCPT |

The mappings are defined in [`src/utils.py`](../src/utils.py).

## Corpora

`Textbooks`, `PubMed`, `StatPearls`, and `Wikipedia` select individual corpora. `MedText` combines Textbooks and StatPearls; `MedCorp` combines all four. The `Guidelines` key is also defined, but this repository does not provide a preparation script for it; prepare compatible local files before selecting it.

See [data and indexing](data-and-indexing.md) for preparation and first-run behavior.

## Dataset example

```bash
python examples/retrieval.py
```

This existing example loads the **MedQA test split**, retrieves 16 passages per question with MedCPT over Textbooks, and prints the top passage for the first question. It performs retrieval across the split, without computing answer accuracy.

[Evidence filtering](evidence-filtering.md) · [Query reformulation](query-reformulation.md)
