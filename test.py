import functools
import csv
import os
from PIL import Image
from huggingface_hub import login
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


# You need your own Hugging Face token to access the model
HF_TOKEN = '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'
print('Loginning in HF')
login(HF_TOKEN)
print('Done')

model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

# CSV_PATH = "/Users/meiyi/Desktop/documents/GT/Spring_2025/Project/data/food_image_recipe.csv"
# IMAGE_PATH = "/Users/meiyi/Desktop/documents/GT/Spring_2025/Project/data/food_images"
CSV_PATH = "./kaggle/clean.csv"
IMAGE_PATH = "kaggle/Food Images/Food Images"
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


if __name__ == '__main__':
    print('Loading CSV')
    examples = load_food_datasets(CSV_PATH)
    # Prepare dataset
    print('Preparing dataset')
    dataset = prepare_dataset(examples)

    # Prepares the model to understand both images and text
    model_kwargs = dict(
        attn_implementation="eager",  # basic attention; use 'flash_attention_2' if a modern GPU
        torch_dtype=torch.bfloat16, # makes calculations faster while maintaining accuracy
        device_map="auto",
    )

    # Quantization configuration
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load model and processor
    print('Loading model')
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    print('Loading processor')
    processor = AutoProcessor.from_pretrained(model_id)

    # PEFT/LoRA configuration: Instead of training the whole massive model, we only tweak small parts
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05, # adds slight randomness to avoid memorizing
        r=16, # how many adjustable points per layer 
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # Training configuration
    args = SFTConfig(
        output_dir="outputs",
        num_train_epochs=1,  # How many times the model sees the entire dataset
        per_device_train_batch_size=1, # How many examples the GPU processes at once
        gradient_accumulation_steps=4, # number of steps before performing a backward/update pass
        gradient_checkpointing=True,
        optim="adamw_torch_fused", # an optimized version of the AdamW optimizer
        logging_steps=5,# prints loss/metrics every 5 steps
        save_strategy="epoch",
        learning_rate=2e-4, # how fast the model learns
        bf16=True,
        max_grad_norm=0.3, # prevents huge updates that could break the model
        warmup_ratio=0.03, # slowly ramps up the learning rate over 3% of training steps
        lr_scheduler_type="cosine", # smoother fine-tuning over many epochs
        push_to_hub=False,
        report_to="tensorboard", # logs metrics to TensorBoard for visualization
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="messages", 
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    args.remove_unused_columns = False


    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
        data_collator=functools.partial(collate_fn, processor=processor),
    )

    # Start training
    print('Start training')
    trainer.train()

    # Save model
    print('Trainig done. Saving')
    trainer.save_model("gemma-recipe-generator-final")
