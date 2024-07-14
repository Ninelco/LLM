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


print(next(gen_batches_train()))

device_map = {"": 0}
model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


training_arguments = TrainingArguments(
    output_dir='/home/ninelcozaurus/project/results/tune_results',
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    optim="adamw_torch",
    save_steps=10,
    logging_steps=5,
    learning_rate=3e-4,
    fp16=False,
    bf16=True,
    num_train_epochs=1,
    report_to="none"
)

train_gen = Dataset.from_generator(gen_batches_train)
tokenizer.padding_side = "right"

trainer = SFTTrainer(
    model=model,
    train_dataset=train_gen,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

peft_model_id="/home/ninelcozaurus/project/results/llama_lora2"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)