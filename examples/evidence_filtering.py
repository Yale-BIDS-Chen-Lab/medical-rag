from transformers import AutoTokenizer, AutoModelForCausalLM

import torch


# Instruction used during training
INSTRUCTION = (
    "Given a query and a text passage, determine whether the passage contains supporting evidence for the query. "
    "Supporting evidence means that the passage provides clear, relevant, and factual information that directly backs or justifies the answer to the query.\n\n"
    "Respond with one of the following labels:\n\"Yes\" if the passage contains supporting evidence for the query.\n"
    "\"No\" if the passage does not contain supporting evidence.\n"
    "You should respond with only the label (Yes or No) without any additional explanation."
)


def main():
    model_id = "Yale-BIDS-Chen/Llama-3.1-8B-Evidence-Filtering"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Example query + retrieved passage
    query = "What is the first-line treatment for acute angle-closure glaucoma?"
    doc = "Acute angle-closure glaucoma requires immediate treatment with topical beta-blockers, alpha agonists, and systemic carbonic anhydrase inhibitors."

    # Build chat-style prompt
    content = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"Question: {query}\nPassage: {doc}"}
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    # Tokenize
    input_ids = tokenizer(content, return_tensors="pt").input_ids.to(model.device)

    # Define stopping tokens (Llama-3 style)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate evidence-filtering judgment
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
    )

    # Decode model response
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))

if __name__ == "__main__":
    main()

