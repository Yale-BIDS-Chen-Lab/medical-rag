# Installation

[Documentation](README.md) / Installation

## Environment

Use Python 3.10 in a dedicated environment. The full dependency set includes vLLM and targets a Linux machine with an NVIDIA GPU. The dense retrieval code can use a CPU, but installing the full requirements on macOS or a CPU-only machine requires a separate, adapted environment.

```bash
git clone https://github.com/Yale-BIDS-Chen-Lab/medical-rag.git
cd medical-rag
conda create -n medical-rag python=3.10 -y
conda activate medical-rag
```

Run all examples from the repository root.

## Dependencies

The pinned `vllm==0.10.0` requires **PyTorch 2.7.1**, with matching torchvision and torchaudio versions. The following uses the corresponding CUDA 12.8 wheels; the NVIDIA driver must support this CUDA runtime.

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install "numpy<2" -r requirements.txt
python -m pip check
```

The NumPy constraint keeps the environment compatible with the pinned `faiss-cpu==1.7.4` binary. If you adapt the environment, check the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/) and [vLLM 0.10.0 dependencies](https://github.com/vllm-project/vllm/blob/v0.10.0/requirements/cuda.txt) together.

The existing corpus/index preparation code also invokes `git`, `wget`, and `unzip`; make these available on your system. BM25 and RRF retrieval additionally require a Java runtime compatible with the pinned [Pyserini](https://github.com/castorini/pyserini) version.

## Check the environment

```bash
python -c "import torch, faiss; from src.utils import RetrievalSystem; print('Retrieval imports OK'); print('CUDA available:', torch.cuda.is_available())"
python examples/quickstart.py --help
```

Then follow the [single-query quick start](../README.md#quick-start). The first actual retrieval downloads corpus files and model weights and prepares an index; `--help` does not download anything.

## Component requirements

| Component | Requirements |
| --- | --- |
| Dense retrieval | Corpus, query encoder, and a local FAISS index; CPU fallback is available |
| BM25 / RRF | Java and Pyserini in addition to the retrieval dependencies |
| Evidence filtering | Download the released 8B model; the example uses bfloat16 and automatic device placement |
| Rationale querying | An NVIDIA GPU suitable for the selected model and vLLM; the example uses Llama-3.1-8B-Instruct |

The retrieval quick start does not use an LLM API. If Hugging Face requests authentication or access approval for a selected model, complete that model's access steps before running its example. Download size and memory needs depend on the selected corpus and model.

[Retrieval guide](retrieval.md) · [Data and indexing](data-and-indexing.md)
