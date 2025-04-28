import csv
import os
from PIL import Image
from datasets import Dataset


# CSV_PATH = "/Users/meiyi/Desktop/documents/GT/Spring_2025/Project/data/food_image_recipe.csv"
# IMAGE_PATH = "/Users/meiyi/Desktop/documents/GT/Spring_2025/Project/data/food_images"
CSV_PATH = "../data/image_recipe_mapping_100.csv"
IMAGE_PATH = "../data/food_images"
SYSTEM_MESSAGE = "You are an expert chef and cooking recipe writer."


# Following the instruction, need to create a prompt template to combine the image and a system message for the assistant.
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

# Takes the food examples (image + recipe) and converts them into a 'message' format that the model can understand
def prepare_dataset(examples):
    dataset = []
    for example in examples:
        try:
            message = convert_to_message(example)
            dataset.append(message)
        except Exception as e:
            print(f"Skipping example {example.get('Image_Name', 'unknown')} due to error: {e}")
    return Dataset.from_list(dataset) # packages everything into a dataset object


# Custom collator for image-text pairs
def collate_fn(examples, processor):
    texts = []
    images = []
    
    for example in examples:
        # Extract the image from messages
        user_content = example["messages"][1]["content"]
        image = user_content[0]["image"]  # Get the PIL Image object
        
        # Generate the chat text
        text = processor.apply_chat_template(
            example["messages"], 
            add_generation_prompt=False, 
            tokenize=False
        )
        texts.append(text.strip())
        images.append(image)

    # Process both text and images together
    batch = processor(
        text=texts, 
        images=images, 
        return_tensors="pt", 
        padding=True,
        truncation=True
    )

    # Prepare labels for training
    labels = batch["input_ids"].clone()
    
    # Mask padding tokens and special tokens in loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Mask image tokens if they exist in the tokenizer
    if hasattr(processor.tokenizer, 'special_tokens_map') and 'boi_token' in processor.tokenizer.special_tokens_map:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
        labels[labels == image_token_id] = -100
    
    batch["labels"] = labels
    return batch
