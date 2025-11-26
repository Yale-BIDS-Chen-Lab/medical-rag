# Repository for the implementation of medical RAG systems

This repository implements a lightweight and modular **Retrieval-Augmented Generation (RAG)** pipeline for **medical and clinical tasks**.  
It is built on top of the [MedRAG toolkit](https://github.com/Teddy-XiongGZ/MedRAG) and adds a small set of practical extensions:

- 🔥 Support for recent dense retrievers (e.g., *Qwen3-embedding*)
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
