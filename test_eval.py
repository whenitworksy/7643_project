import functools
import csv
import os
from PIL import Image
from huggingface_hub import login
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, DatasetDict
from evaluate import load
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from meteor_score import meteor_score
import spacy

# Initialize spaCy for ingredient parsing
nlp = spacy.load("en_core_web_sm")

HF_TOKEN = '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'
SYSTEM_MESSAGE = "You are an expert chef and cooking recipe writer."
CSV_PATH = "./kaggle/clean.csv"
IMAGE_PATH = "kaggle/Food Images/Food Images"

print('Logging in HF')
login(HF_TOKEN)
print('Done')

model_id = "google/gemma-3-4b-pt"


def convert_to_message(example):
    try:
        image = Image.open(os.path.join(IMAGE_PATH, f'{example["Image_Name"]}.jpg'))
    except FileNotFoundError:
        raise ValueError(f"Missing image: {example['Image_Name']}")

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text",
                             "text": f"Title: {example['Title']}\nIngredients:\n{example['Ingredients']}\nInstructions:\n{example['Instructions']}"}],
            }
        ]
    }


def load_food_datasets(csv_path: str):
    outputs = []
    with open(csv_path, 'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            del row['']
            outputs.append(row)
    return outputs


def prepare_dataset(examples):
    full_dataset = Dataset.from_list([convert_to_message(ex) for ex in examples])

    # First split: 80% train, 20% temp
    train_test_split = full_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

    # Second split: 20% temp -> 10% val, 10% test
    val_test_split = train_test_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)

    return DatasetDict({
        'train': train_test_split['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })


def collate_fn(examples, processor):
    texts = []
    images = []

    for example in examples:
        user_content = example["messages"][1]["content"]
        image = user_content[0]["image"]

        text = processor.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False
        )
        texts.append(text.strip())
        images.append(image)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

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

    print('Preparing dataset splits')
    dataset = prepare_dataset(examples)

    print(f"Train: {len(dataset['train'])} samples")
    print(f"Validation: {len(dataset['validation'])} samples")
    print(f"Test: {len(dataset['test'])} samples")

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    )

    print('Loading model')
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    print('Loading processor')
    processor = AutoProcessor.from_pretrained(model_id)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    args = SFTConfig(
        output_dir="outputs",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="messages",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
        data_collator=functools.partial(collate_fn, processor=processor),
    )

    print('Start training')
    trainer.train()

    print('\nRunning final evaluation on test set...')
    test_results = trainer.evaluate(dataset['test'])

    # Initialize metrics
    bleu = BLEU()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    title_bleu_scores = []
    ingredient_bleu_scores = []
    instruction_meteor_scores = []
    overall_rouge_scores = []

    for example in dataset['test']:
        try:
            # Generate prediction (replace with actual model inference)
            predicted_recipe = "Sample prediction\nIngredients:\n- flour\n- sugar\nInstructions:\nMix ingredients and bake"

            # Parse components
            ref_content = example["messages"][2]["content"][0]["text"]
            ref_title = ref_content.split('\n')[0].replace('Title:', '').strip()
            ref_ingredients = ' '.join([line.replace('-', '').strip()
                                        for line in
                                        ref_content.split('Ingredients:')[1].split('Instructions:')[0].split('\n')
                                        if line.strip()])
            ref_instructions = ref_content.split('Instructions:')[1].strip()

            # Calculate metrics
            title_bleu_scores.append(bleu.sentence_score(predicted_recipe.split('\n')[0], [ref_title]).score)

            pred_ingredients = ' '.join([line.replace('-', '').strip()
                                         for line in
                                         predicted_recipe.split('Ingredients:')[1].split('Instructions:')[0].split('\n')
                                         if line.strip()])
            ingredient_bleu = bleu.corpus_score([pred_ingredients.split()], [[ref_ingredients.split()]])
            ingredient_bleu_scores.append(ingredient_bleu.score)

            pred_instructions = predicted_recipe.split('Instructions:')[1].strip()
            meteor_score_val = meteor_score([ref_instructions], pred_instructions)
            instruction_meteor_scores.append(meteor_score_val)

            overall_rouge = rouge.score(predicted_recipe, ref_content)['rougeL'].fmeasure
            overall_rouge_scores.append(overall_rouge)

        except Exception as e:
            print(f"Skipping example due to error: {e}")
            continue

    print(f"\nFinal Metrics:")
    print(f"Title BLEU: {sum(title_bleu_scores) / len(title_bleu_scores):.2f}")
    print(f"Ingredient BLEU: {sum(ingredient_bleu_scores) / len(ingredient_bleu_scores):.2f}")
    print(f"Instruction METEOR: {sum(instruction_meteor_scores) / len(instruction_meteor_scores):.2f}")
    print(f"Overall ROUGE-L: {sum(overall_rouge_scores) / len(overall_rouge_scores):.2f}")

    print('\nTraining done. Saving model...')
    trainer.save_model("gemma-recipe-generator-final")
