import csv
import os
from PIL import Image
from huggingface_hub import login
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# You need your own Hugging Face token to access the model
HF_TOKEN = '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'
login(HF_TOKEN)

model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

CSV_PATH = "data/food_image_recipe_1k.csv"
IMAGE_PATH = "data/food_images"
SYSTEM_MESSAGE = "You are an expert chef and cooking recipe writer."

# processor = AutoProcessor.from_pretrained(model_id)

# Following the instruction, you need to create a prompt template to combine the image and a system message for the assistant.
def convert_to_message(example):
    image = Image.open(os.path.join(IMAGE_PATH, f'{example["Image_Name"]}.jpg'))
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", # prompt
                        "image": image,
                    },
                ]
            }
        ],
    }

def load_food_datasets(csv_path: str):
    outputs = []
    with open(csv_path, 'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            del row['']
            outputs.append(row)
    return outputs


def collate_fn(examples):
    pass

if __name__ == '__main__':
    examples = load_food_datasets(CSV_PATH)
    print(len(examples))
    for i, example in enumerate(examples[:5]):
        print(f'========Example {i} ')
        for k, v in example.items():
            print(f'---{k}: {v[:50]}')
        print('Converting to message')
        msg = convert_to_message(example)
        print(msg)
