from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, \
    GenerationConfig
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model

from trl import SFTTrainer

print(torch.cuda.is_available())
print(torch.cuda.current_device())

import json

with open('/home/ninelcozaurus/project/dataset/train.json', 'r') as f:
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

dataframe = pd.DataFrame(data)
print(dataframe.head(100))

dataset = Dataset.from_pandas(dataframe)
print(dataset)

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(tokenizer.pad_token, tokenizer.eos_token)
tokenizer.pad_token = tokenizer.eos_token

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)


def gen_batches_train():
    for sample in iter(dataset):
        # Extract instruction and input from the sample
        system_prompt = "You are a professional examiner with a deep knowledge of the subject. Your task is to help you compose questions for the student exam."
        input_text = f"# Question: {sample['question']}\n# Right answer: {sample['correct_answer']}\n\nCreate 3 plausible but incorrect answers (distractors) for this question. Generate 3 incorrect answers (distractors) in the following format:\n# Distractors:\n - <wrong answer 1>\n - <wrong answer 2>\n - <wrong answer 3>.\nDon't add numbers or letters to the answers."
        out_text = f"# Distractors:\n - {sample['distractor_1']}\n - {sample['distractor_2']}\n - {sample['distractor_3']}"
        formatted_prompt = None

        formatted_prompt = tokenizer.apply_chat_template([{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": input_text
        }, {
            "role": "assistant",
            "content": out_text
        }], tokenize=False, add_generation_prompt=False) + '<|end_of_text|>'

        yield {'text': formatted_prompt}


next(gen_batches_train())