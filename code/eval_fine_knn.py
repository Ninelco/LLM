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
with open('/home/ninelcozaurus/project/dataset/val.json', 'r') as f:
    file = json.load(f)

data = []

for dict in file:
    question = dict['question']
    distractor_1 = dict['distractor1']
    distractor_2 = dict['distractor2']
    distractor_3 = dict['distractor3']

    data.append(
        {
            'question': question,
            'distractor_1': distractor_1,
            'distractor_2': distractor_2,
            'distractor_3': distractor_3
        }
    )

df = pd.DataFrame(data)
print(df)
ds = Dataset.from_pandas(df)


# Function to generate with prompt
# def generate_with_prompt(model, tokenizer, dataset):
#     model.eval()
#     with torch.no_grad():
#         for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
#             system_prompt = "You are a professional examiner with a deep knowledge of the subject. Your task is to help you compose questions for the student exam."
#             input_text = f"# Question: {sample['question']}\n\nCreate 3 plausible but incorrect answers (distractors) for this question. Generate 3 incorrect answers (distractors) in the following format:\n# Distractors:\n - <wrong answer 1>\n - <wrong answer 2>\n - <wrong answer 3>.\nDon't add numbers or letters to the answers."

#             formatted_prompt = tokenizer.apply_chat_template([{
#                 "role": "system",
#                 "content": system_prompt
#             }, {
#                 "role": "user",
#                 "content": input_text
#             }], tokenize=False, add_generation_prompt=True)

#             print("INPUT:")
#             print('\n***************---------------------**************\n')
#             print(formatted_prompt)

#             inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, padding=True, max_length=1024)
#             inputs = {key: val.to(model.device) for key, val in inputs.items()}
#             labels = inputs["input_ids"].clone()
#             print(f"labels = {labels}")
#             outputs = model(**inputs, labels=labels)
#             print("OUTPUT")
#             print(outputs)
#             print('\n*********************************************************\n\n\n')

with open("/home/ninelcozaurus/project/results/output_fine_distract.txt", 'w') as f:
    f.write(f"Start output" + '\n')

def generate_with_prompt(model, tokenizer, dataset):
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
                # Формирование текстового промпта
                system_prompt = "You are a professional examiner with a deep knowledge of the subject. Your task is to help you compose questions for the student exam."
                input_text = f"# Question: {sample['question']}\n\nCreate 3 plausible but incorrect answers (distractors) for this question. Generate 3 incorrect answers (distractors) in the following format:\n# Distractors:\n - <wrong answer 1>\n - <wrong answer 2>\n - <wrong answer 3>.\nDon't add numbers or letters to the answers."

                formatted_prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ], tokenize=False, add_generation_prompt=True)

                print("INPUT:")
                print('\n***************---------------------**************\n')
                print(formatted_prompt)

                with open("/home/ninelcozaurus/project/results/output_fine_distract.txt", 'a') as f:
                    f.write(f"INPUT:" + '\n')
                    f.write(f'\n***************---------------------**************\n{formatted_prompt}')

                # Подготовка входных данных и меток
                inputs = tokenizer(formatted_prompt, return_tensors='pt', truncation=True, padding=True, max_length=1024)
                inputs = {key: val.to(model.device) for key, val in inputs.items()}
                labels = inputs["input_ids"].clone()

                # Декодирование меток
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Вывод декодированных меток
                print(f"Decoded labels:")
                for label in decoded_labels:
                    print(label)
                
                with open("/home/ninelcozaurus/project/results/output_fine_distract.txt", 'a') as f:
                    f.write(f"Decoded labels:" + '\n')
                    f.write(f'\n---------------------------------------------------------------\n{decoded_labels}')
                
                # # Получение выходных данных модели
                # outputs = model(**inputs, labels=labels)
                # # print(outputs)
                # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Получение выходных данных модели
                # outputs = model(**inputs, labels=labels)
                # Andreas
                outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                # logits = outputs.logits

                # Получение ID предсказанных токенов
                # predicted_ids = torch.argmax(logits, dim=-1)
                
                # Декодирование предсказанных токенов
                # decoded_output = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

                # Andreas 
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print("\n\nDecoded_output:")
                print(generated_text)
                print('\n*********************************************************\n\n\n')
                # with open("/home/ninelcozaurus/project/results/output_fine_distract.txt", 'a') as f:
                #     f.write(f"OUTPUT" + '\n')
                #     f.write(f'\n*********************************************************\n\n\n{decoded_output}')

                # Запись выходных данных модели в файл
                with open("/home/ninelcozaurus/project/results/output_fine_distract.txt", 'a') as f:
                    f.write(f"\n\nOUTPUT:\n")
                    f.write(f'\n*********************************************************\n')
                    f.write(f"{generated_text}\n")
            


# Load original models
fine_tuned_model_path = '/home/ninelcozaurus/project/results/llama_lora2'
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

# Select a subset for evaluation
eval_dataset = ds

# Compute loss for each model
fine_tuned_model_loss = generate_with_prompt(fine_tuned_model, fine_tuned_tokenizer, eval_dataset)

# with open("/home/ninelcozaurus/project/results/loss.txt", 'w') as f:
#     f.write(f"FineTuned Llama 3 8B Model with fine tune have Loss: {fine_tuned_model_loss}" + '\n')

# print(f"FineTuned Llama 3 8B Model have Loss: {fine_tuned_model_loss}")
