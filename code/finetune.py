from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch

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

df = pd.DataFrame(data)
print(df.head(100))

ds = Dataset.from_pandas(df)
print(ds)