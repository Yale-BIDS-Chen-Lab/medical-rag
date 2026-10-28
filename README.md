<p align="center">
  <img src="assets/wordmark.svg" alt="Medical RAG" width="440">
</p>

<p align="center"><strong>Tools for evidence-grounded medical AI</strong></p>

<p align="center">
  <a href="#research">📄 Research</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200">📊 Dataset</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering">🤗 Model</a> &nbsp;·&nbsp;
  <a href="#quick-start">⚡ Quick Start</a> &nbsp;·&nbsp;
  <a href="docs/README.md">📚 Docs</a>
</p>

Research code and reusable components for retrieval-augmented generation in medicine, including evidence retrieval, evidence filtering, and query reformulation.

<p align="center">
  <img src="assets/overview.svg" alt="Medical RAG capabilities: 1 query reformulation, 2 evidence retrieval, 3 evidence filtering, and 4 expert evaluation data. The query-to-response flow uses retrieval and generation, with optional reformulation and filtering." width="100%">
</p>

## Highlights

| Capability | Explore |
| --- | --- |
| **① Query reformulation**<br>Rationale-based queries for medical evidence retrieval. | [Guide](docs/query-reformulation.md) · [Example](examples/retrieval_using_rationale_query.py) |
| **② Evidence retrieval**<br>Sparse, dense, and fusion retrieval with BM25, MedCPT, and Qwen3. | [Guide](docs/retrieval.md) · [Example](examples/quickstart.py) |
| **③ Evidence filtering**<br>A released Llama-3.1-8B-based query–passage classifier. | [Guide](docs/evidence-filtering.md) · [Model](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering) |
| **④ Expert evaluation data**<br>Labels for evidence retrieval, selection, factuality, and completeness. | [Dataset](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200) · [Guide](docs/research/rethinking-med-rag.md#released-evaluation-data) |

## Research

| Study | Resources |
| --- | --- |
| **[Rethinking Medical RAG](docs/research/rethinking-med-rag.md)**<br>Expert evaluation of evidence retrieval, selection, and response quality.<br>**200 questions · 800 responses · 3,200 passages** | [📄 Paper](https://arxiv.org/abs/2511.06738)<br>[📊 Dataset](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200) · [🤗 Model](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering) |

[Research guides and code maps →](docs/research/README.md)

## Quick Start

Clone the repository and follow the [installation guide](docs/installation.md), then run a single-query retrieval example from the repository root:

```bash
python examples/quickstart.py \
  --query "What is the role of insulin in glucose metabolism?" \
  --k 3
```

This searches the **Textbooks** corpus with **MedCPT** and prints ranked passages with their titles, text, and retrieval scores. It does not generate an answer or require an LLM API key.

On the first run, the retriever downloads the corpus and model weights and prepares a local index. This requires network access and disk space; later runs reuse the saved files. See [data and indexing](docs/data-and-indexing.md) for the directory layout and preparation details.

<details>
<summary>Output format</summary>

The following is a format illustration, not a recorded retrieval result. Passage text and scores depend on the corpus and index.

```text
Query: What is the role of insulin in glucose metabolism?
Retriever: MedCPT | Corpus: Textbooks
Retrieved passages: 3

[1] <passage title>
ID: <passage id> | Score: <retrieval score>
<passage text>
```

</details>

For the available options, run `python examples/quickstart.py --help` or see the [retrieval guide](docs/retrieval.md).

## Resources

The dataset and filtering model are available in the [Medical RAG collection on Hugging Face](https://huggingface.co/collections/Yale-BIDS-Chen/medical-rag).

| Resource | Description | Link |
| --- | --- | --- |
| Expert evaluation dataset | Questions, responses, retrieved passages, and evaluation labels | [Hugging Face](https://huggingface.co/datasets/Yale-BIDS-Chen/medical-rag-expert-evaluation-200) |
| Evidence filtering model | Llama-3.1-8B-based query–passage evidence classifier | [Hugging Face](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering) |

## Documentation

- [Installation](docs/installation.md) — environment, dependencies, and hardware notes
- [Data and indexing](docs/data-and-indexing.md) — corpus format, embeddings, and local indexes
- [All guides](docs/README.md) — component examples and research documentation

```text
medical-rag/
├── assets/        README graphics
├── docs/          Component guides and research documentation
├── examples/      Runnable component examples
├── src/           Retrieval, generation, prompts, and corpus preparation
└── requirements.txt
```

## Citation

If you use this work, please cite:

```bibtex
@article{kim2025rethinking,
  title={Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights},
  author={Kim, Hyunjae and Sohn, Jiwoong and Gilson, Aidan and Cochran-Caggiano, Nicholas and Applebaum, Serina and Jin, Heeju and Park, Seihee and Park, Yujin and Park, Jiyeong and Choi, Seoyoung and others},
  journal={arXiv preprint arXiv:2511.06738},
  year={2025}
}
```

## Acknowledgements

The initial RAG implementation was adapted from the [MedRAG toolkit](https://github.com/Teddy-XiongGZ/MedRAG). We thank its authors and the developers of the models, datasets, and libraries used in this project.

## Disclaimer

The information produced by this repository is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced by this repository.

## Contact

For questions or suggestions, [open an issue](https://github.com/Yale-BIDS-Chen-Lab/medical-rag/issues) or contact [Hyunjae Kim](mailto:hyunjae.kim@yale.edu) or [Qingyu Chen](mailto:qingyu.chen@yale.edu).
