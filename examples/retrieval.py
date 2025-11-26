
'''
Example code for evidence retrieval.
Given a set of questions, the specified retriever searches the specified corpus and appends the retrieved evidence to a list. 
In this example, we use the MedQA dataset as the question set, MedCPT as the retriever, and a textbook corpus. 
For available retriever and corpus options, please refer to src/utils.py.
'''

import json
import os
import sys

from datasets import load_dataset
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.utils import RetrievalSystem

def load_medqa():
    questions, labels = [],[]
    for sample in load_dataset(
        "GBaker/MedQA-USMLE-4-options", split="test", trust_remote_code=True
    ):
        options = sorted(sample["options"].items())
        options = " ".join(map(lambda x: f"({x[0]}) {x[1]}", options))

        question = sample["question"] + " " + options
        
        questions.append(question)
        labels.append(sample["answer_idx"])

    return questions, labels


def main():
    queries, _ = load_medqa()

    retriever = RetrievalSystem(retriever_name="MedCPT", corpus_name="Textbooks", cache=True)

    all_snippets = []
    for question in tqdm(queries):
        snippets, _ = retriever.retrieve(question=question, k=16)
        all_snippets.append(snippets)
    
    print("\nEvidence retrieval for the {} queries has been completed.".format(str(len(all_snippets))))
    print("\n## Shown here is the top-1 retrieved passage for the first query.")
    print("\n### Query:\n", queries[0])
    print("\n### Retrieved Passage:\n", all_snippets[0][0])

if __name__ == "__main__":
    main()

