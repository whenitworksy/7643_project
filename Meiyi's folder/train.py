import functools

import data_loader

from huggingface_hub import login
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig # replace AutoModelForImageTextToText with AutoModelForVision2Seq
import torch
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split


# You need your own Hugging Face token to access the model
HF_TOKEN = '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'
print('Loginning in HF')
login(HF_TOKEN)
print('Done')
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42  # For reproducibility

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

def split_dataset(dataset, test_size=0.2, random_state=42):
    """Split dataset into train and test sets"""
    # Convert dataset to list for splitting
    dataset_list = [item for item in dataset]
    
    # Split indices
    train_idx, test_idx = train_test_split(
        range(len(dataset_list)),
        test_size=test_size,
        random_state=random_state
    )
    
    # Create train/test datasets
    train_data = [dataset_list[i] for i in train_idx]
    test_data = [dataset_list[i] for i in test_idx]
    
    return train_data, test_data

if __name__ == '__main__':
    print('Loading CSV')
    examples = data_loader.load_food_datasets(CSV_PATH)
    # Prepare dataset
    print('Preparing dataset')
    dataset = data_loader.prepare_dataset(examples)

    # Split dataset
    print('Splitting dataset')
    train_dataset, test_dataset = split_dataset(dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')


    # Prepares the model to understand both images and text
    model_kwargs = dict(
        attn_implementation="eager",  # basic attention; use 'flash_attention_2' if a modern GPU
        torch_dtype=torch.bfloat16, # makes calculations faster while maintaining accuracy
        device_map="auto",
    )

    # # Quantization configuration
    # model_kwargs["quantization_config"] = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_storage=torch.bfloat16,
    #     llm_int8_enable_fp32_cpu_offload=True,
    # )

    # Model configuration - UPDATED FOR SMOLVLM
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # SmolVLM is smaller, may not need all these quantization settings
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    )

    # Load model and processor, updated fpr SMOLVLM
    print('Loading model')
    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_kwargs)
    print('Loading processor')
    processor = AutoProcessor.from_pretrained(model_id, size={"longest_edge": 512})

    # # PEFT/LoRA configuration: Instead of training the whole massive model, we only tweak small parts
    # peft_config = LoraConfig(
    #     lora_alpha=16,
    #     lora_dropout=0.05, # adds slight randomness to avoid memorizing
    #     r=16, # how many adjustable points per layer 
    #     bias="none",
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    #     modules_to_save=["lm_head", "embed_tokens"],
    # )

    # # Training configuration
    # args = SFTConfig(
    #     output_dir="outputs",
    #     num_train_epochs=1,  # How many times the model sees the entire dataset
    #     per_device_train_batch_size=1, # How many examples the GPU processes at once
    #     gradient_accumulation_steps=4, # number of steps before performing a backward/update pass
    #     gradient_checkpointing=True,
    #     optim="adamw_torch_fused", # an optimized version of the AdamW optimizer
    #     logging_steps=1,# prints loss/metrics every 5 steps
    #     save_strategy="epoch",
    #     learning_rate=2e-4, # how fast the model learns
    #     bf16=True,
    #     max_grad_norm=0.3, # prevents huge updates that could break the model
    #     warmup_ratio=0.03, # slowly ramps up the learning rate over 3% of training steps
    #     lr_scheduler_type="cosine", # smoother fine-tuning over many epochs
    #     push_to_hub=False,
    #     report_to="tensorboard", # logs metrics to TensorBoard for visualization
    #     gradient_checkpointing_kwargs={"use_reentrant": False},
    #     dataset_text_field="", 
    #     dataset_kwargs={"skip_prepare_dataset": True},
    # )
    # args.remove_unused_columns = False


        # PEFT/LoRA configuration - ADJUSTED FOR SMOLVLM
    peft_config = LoraConfig(
        lora_alpha=8,  # Reduced from 16 for smaller model
        lora_dropout=0.05,
        r=8,  # Reduced from 16
        bias="none",
        target_modules=["q_proj", "v_proj"],  # More specific target modules
        task_type="CAUSAL_LM",
    )

    # Training configuration - ADJUSTED FOR SMOLVLM
    args = SFTConfig(
        output_dir="outputs",
        num_train_epochs=3,  # Increased from 1 as model is smaller
        per_device_train_batch_size=2,  # Increased from 1
        gradient_accumulation_steps=2,  # Reduced from 4
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=1e-4,  # Reduced from 2e-4
        bf16=True,
        max_grad_norm=0.5,  # Increased from 0.3
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="messages",  # Changed from empty string
        dataset_kwargs={"skip_prepare_dataset": False},  # Changed from True
    )


    # Initialize trainer with train dataset only
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=functools.partial(data_loader.collate_fn, processor=processor),
    )

    # Start training
    print('Start training')
    trainer.train()

    # Save model
    print('Trainig done. Saving')
    trainer.save_model("smolvom-recipe-generator-final")
