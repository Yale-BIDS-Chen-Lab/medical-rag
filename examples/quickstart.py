"""Retrieve passages for a single query without running a benchmark or an LLM."""

import argparse
from pathlib import Path
import sys


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What is the role of insulin in glucose metabolism?",
        help="Question used to retrieve passages.",
    )
    parser.add_argument("--k", type=positive_int, default=3, help="Number of passages.")
    parser.add_argument("--retriever", default="MedCPT", help="Retriever name (default: MedCPT).")
    parser.add_argument("--corpus", default="Textbooks", help="Corpus name (default: Textbooks).")
    parser.add_argument("--db-dir", default="./corpus", help="Local corpus/index directory.")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must contain a non-empty question")
    return parser, args


def main():
    parser, args = parse_args()

    # Keep --help usable before installing model and retrieval dependencies.
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from src.utils import RetrievalSystem, corpus_names, retriever_names

    if args.retriever not in retriever_names:
        parser.error("--retriever must be one of: " + ", ".join(retriever_names))
    if args.corpus not in corpus_names:
        parser.error("--corpus must be one of: " + ", ".join(corpus_names))

    print("Preparing retrieval; the first run may download corpus files and model weights.", flush=True)
    retriever = RetrievalSystem(
        retriever_name=args.retriever,
        corpus_name=args.corpus,
        db_dir=args.db_dir,
        cache=False,
    )
    passages, scores = retriever.retrieve(question=args.query.strip(), k=args.k)

    print(f"\nQuery: {args.query.strip()}")
    print(f"Retriever: {args.retriever} | Corpus: {args.corpus}")
    print(f"Retrieved passages: {len(passages)}")
    for rank, (passage, score) in enumerate(zip(passages, scores), start=1):
        print(f"\n[{rank}] {passage.get('title', '')}")
        print(f"ID: {passage.get('id', '')} | Score: {float(score):.4f}")
        print(passage.get("content", ""))


if __name__ == "__main__":
    main()
