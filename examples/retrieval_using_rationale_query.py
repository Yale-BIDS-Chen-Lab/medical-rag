
'''
Example code for rationale querying.
Given a set of questions, the retriever first reformulates each question into a rationale-style query, which corresponds to the model's own intermediate answer (rationale) to the initial question.
This reformulated rationale query is then used to search the specified corpus.
For the motivation behind rationale-based retrieval and detailed explanations of this approach, please refer to the references below.
- Kim et al., Rethinking Retrieval-Augmented Generation for Medicine: A Large-Scale, Systematic Expert Evaluation and Practical Insights, 2025
- Sohn et al., Rationale-Guided Retrieval Augmented Generation for Medical Question Answering, 2025

After retrieval, the resulting passages can be appended to a list for downstream RAG pipelines or further analysis.
In this example, we use the MedQA dataset as the question set, apply rationale-query reformulation, use MedCPT as the retriever, and retrieve from a textbook corpus.
For available retriever and corpus options, please refer to src/utils.py.
'''

import json
import os
import sys

from vllm import LLM, SamplingParams
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


def generate(model_name_or_path, queries):
    PROMPT = (
        "Answer to the provided multiple-choice question about medical knowledge in a step-bystep fashion. "
        "Output your explanation and single option from the given options as the final answer."
    )
    
    llm = LLM(
        model=model_name_or_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=128000,
        trust_remote_code=True,
        tensor_parallel_size=1  # Two GPUs are required for 70B models
    )
    
    tokenizer = llm.get_tokenizer()

    inputs = []
    for question in queries:
        content = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT + "\n\n### Input Query\n\n" + question}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs.append(content)
    
    generated = llm.generate(
        inputs,
        SamplingParams(
            temperature=0.0,
            repetition_penalty=1.1,
            stop_token_ids=[tokenizer.vocab["<|eot_id|>"]],
            max_tokens=4096,
        ),
    )

    outputs = [g.outputs[0].text for g in generated]
    
    assert len(outputs) == len(queries)

    return outputs


def main():
    initial_queries, _ = load_medqa()
    
    # Reformulate the initial queries using Llama-3.1
    model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    reformulated_queries = generate(model_name_or_path, initial_queries)
    
    # Perform retrieval using the reformulated queries
    retriever = RetrievalSystem(retriever_name="MedCPT", corpus_name="Textbooks", cache=True)
    
    all_snippets = []
    for question in tqdm(reformulated_queries):
        snippets, _ = retriever.retrieve(question=question, k=16)
        all_snippets.append(snippets)
    
    print("\nEvidence retrieval for the {} queries has been completed.".format(str(len(all_snippets))))
    print("\n## Shown here is the top-1 retrieved passage for the first query.")
    print("\n### Initial Query:\n", initial_queries[0])
    print("\n### Reformulated Query (Rationale):\n", reformulated_queries[0])
    print("\n### Retrieved Passage:\n", all_snippets[0][0])

if __name__ == "__main__":
    main()

