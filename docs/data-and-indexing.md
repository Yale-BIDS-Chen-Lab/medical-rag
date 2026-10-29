# Data and Indexing

[Documentation](README.md) / Data and Indexing

Corpus preparation, passage embeddings, and search indexes are shared by the retrieval examples. The current implementation is in [`src/utils.py`](../src/utils.py), with corpus preparation scripts under [`src/data/`](../src/data).

## First-run preparation

When `RetrievalSystem` initializes a corpus and retriever:

1. If the corpus's `chunk/` directory is missing, the code attempts to clone the corresponding `MedRAG/<corpus>` dataset from Hugging Face.
2. For StatPearls, the code also downloads source content from NCBI and invokes the preparation script.
3. For dense retrieval, it loads an existing FAISS index or prepares embeddings and builds an index. Some MedRAG corpus/encoder combinations use precomputed embeddings downloaded from upstream URLs.
4. For BM25, it loads or builds a Pyserini/Lucene index.

These operations can download substantial files. Check storage, network access, and the upstream data requirements before selecting a large corpus. The existing preparation code invokes system tools listed in the [installation guide](installation.md).

## Local layout

By default, retrieval stores files under `./corpus`:

```text
corpus/
└── textbooks/
    ├── chunk/
    │   └── <source>.jsonl
    └── index/
        └── ncbi/
            └── MedCPT-Article-Encoder/
                ├── embedding/
                ├── metadatas.jsonl
                └── faiss.index
```

Each corpus and encoder has its own index path. Pass `db_dir` to `RetrievalSystem` or `--db-dir` to the quick-start example to use another base directory.

## Passage format

Each line of a chunk file contains a JSON object. For example:

```json
{"id":"source_0","title":"Example document","content":"Example passage text.","contents":"Example document. Example passage text."}
```

- `id`: a unique passage identifier
- `title`: the source title
- `content`: passage text used by dense encoders and returned to the caller
- `contents`: the combined searchable text used for BM25 indexing

The field values above are a format illustration. Keep the passage IDs unique across the corpus. Preparation scripts for PubMed, StatPearls, Textbooks, and Wikipedia are provided in [`src/data/`](../src/data); they expect their respective source files and paths.

## Embeddings and indexes

`embed(...)` saves passage embeddings as NumPy arrays. `construct_index(...)` builds a FAISS index and writes metadata linking index positions to chunk files. Dense retrieval uses the configured query encoder to search this index. The `HNSW` constructor option selects an HNSW index when creating a new dense index; otherwise a flat index is built.

The current implementation reuses indexes by file path and does not automatically invalidate them when data or settings change. Use a separate `db_dir` for a different corpus snapshot or index configuration. Keep chunk files consistent with the embeddings and metadata that were built from them.

For reproducible experiments, record the corpus snapshot, chunking settings, encoder revision, and index settings alongside the experiment. New corpus names also need an entry in `corpus_names` in `src/utils.py`; adding a data directory alone does not register a new corpus.

[Retrieval guide](retrieval.md) · [Research guide](research/rethinking-med-rag.md)
