# Repository for the implementation of medical RAG systems

This repository implements a lightweight and modular **Retrieval-Augmented Generation (RAG)** pipeline for **medical and clinical tasks**.  
It is built on top of the [MedRAG toolkit](https://github.com/Teddy-XiongGZ/MedRAG) and adds a small set of practical extensions:

- 🔥 Support for recent dense retrievers (e.g., [*Qwen3-embedding*](https://huggingface.co/collections/Qwen/qwen3-embedding))
- 🔥 Evidence filtering
- 🔥 Query reformulation

The goal is to provide a simple and configurable reference pipeline for experimenting with RAG in the medical domain, while keeping the codebase easy to understand and extend.

---

## 🔧 Installation

### 1. Create environment
```bash
conda create -n medical-rag python==3.10
conda activate medical-rag
```

### 2. Install PyTorch (CUDA version may vary)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

### 3. Install remaining dependencies
pip install -r requirements.txt

## ⚡ Quick Start

This repository provides core components for building RAG systems in the medical domain.  
Each component has a minimal runnable example under the **`examples/`** directory.

### Retrieval
Retrieves top-k candidate passages from a medical corpus using sparse/dense embeddings  
(e.g., *BM25*, *MedCPT*, *Qwen3-embedding*, etc.).

- Input: user query  
- Output: list of retrieved passages (+ scores)
- Example: `examples/retrieval.py`

### Evidence Filtering
Removes irrelevant or low-quality passages from the retrieved set. 
This step is particularly helpful when top-k retrieval includes substantial noise, which is a common issue in clinical and biomedical corpora.

Our implementation uses a fine-tuned evidence-filtering model (based on Llama-3.1-8B-Instruct), trained to judge whether a candidate passage contains supporting evidence for a given medical query.
The model outputs *“Yes”* (contains evidence) or *“No”* (does not contain evidence).
Model weights are publicly available on the [Hugging Face repository](https://huggingface.co/Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering).

- Input: query-passage pair  
- Output: Yes or No  
- Example: `examples/evidence_filtering.py`

### Retrieval with Query Reformulation (Rationale Querying)
Rewrites the original query into a rationale-style query, which corresponds to the model's intermediate reasoning or justification for the answer.
Instead of retrieving evidence using the possibly short or ambiguous initial query, the system uses a richer, more informative rationale-based query, often leading to more accurate and semantically aligned retrieval.

- Input: original user query  
- Output: rationale-style reformulated query  
- Example: `examples/retrieval_using_rationale_query.py`

For the motivation behind rationale-based retrieval and further methodological details, please refer to [this reference](https://aclanthology.org/2025.naacl-long.635/).

## 🙌 Citation

If this repository is useful for your work, please cite:

### Primary Project
```bash
@article{kim2025rethinking,
  title={Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights},
  author={Kim, Hyunjae and Sohn, Jiwoong and Gilson, Aidan and Cochran-Caggiano, Nicholas and Applebaum, Serina and Jin, Heeju and Park, Seihee and Park, Yujin and Park, Jiyeong and Choi, Seoyoung and others},
  journal={arXiv preprint arXiv:2511.06738},
  year={2025}
}
```

### MedRAG Toolkit (Foundation of This Repository)
```bash
@inproceedings{xiong-etal-2024-benchmarking,
    title = "Benchmarking Retrieval-Augmented Generation for Medicine",
    author = "Xiong, Guangzhi  and
      Jin, Qiao  and
      Lu, Zhiyong  and
      Zhang, Aidong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    pages = "6233--6251"
}
```

## 📬 Contact

For questions or suggestions, feel free to open an issue or email us (`hyunjae.kim@yale.edu` or `qingyu.chen@yale.edu`).
