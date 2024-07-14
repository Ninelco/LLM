import sys
import importlib.util
import json
import torch
import pandas as pd

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig
from tqdm import tqdm

# Load dataset
with open('/home/ninelcozaurus/project/dataset/valid.json', 'r') as f:
    file = json.load(f)

data = []

for dict in file:
    question = dict['question']
    correct_answer = dict['correct_answer']
    distractor_1 = dict['distractor1']
    distractor_2 = dict['distractor2']
    distractor_3 = dict['distractor3']

    data.append(
        {
            'question': question,
            'correct_answer': correct_answer,
            'distractor_1': distractor_1,
            'distractor_2': distractor_2,
            'distractor_3': distractor_3
        }
    )

df = pd.DataFrame(data)
ds = Dataset.from_pandas(df)


# Function to compute the loss with prompt
def compute_loss_with_prompt(model, tokenizer, dataset):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
            system_prompt = "You are a professional examiner with a deep knowledge of the subject. Your task is to help you compose questions for the student exam."
            input_text = f"# Question: {sample['question']}\n# Right answer: {sample['correct_answer']}\n\nCreate 3 plausible but incorrect answers (distractors) for this question. Generate 3 incorrect answers (distractors) in the following format:\n# Distractors:\n - <wrong answer 1>\n - <wrong answer 2>\n - <wrong answer 3>.\nDon't add numbers or letters to the answers."

            formatted_prompt = tokenizer.apply_chat_template([{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": input_text
            }], tokenize=False, add_generation_prompt=True)

            print("INPUT:")
            print(formatted_prompt)

            inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, padding=True, max_length=1024)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}
            labels = inputs["input_ids"].clone()
            print(f"labels = {labels}")
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
            total_count += 1
    return total_loss / total_count


# Load original models
llama_8b_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map="auto",
                                                      torch_dtype=torch.bfloat16)

llama_8b_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
llama_8b_tokenizer.pad_token = llama_8b_tokenizer.eos_token

# Select a subset for evaluation
eval_dataset = ds.select(range(100))

# Compute loss for each model
llama_8b_loss = compute_loss_with_prompt(llama_8b_model, llama_8b_tokenizer, eval_dataset)

with open("/home/ninelcozaurus/project/results/loss.txt", 'a') as f:
    f.write(f"Baseline Llama 3 8B Model have Loss: {llama_8b_loss}" + '\n')

print(f"Baseline Llama 3 8B Model have Loss: {llama_8b_loss}")
