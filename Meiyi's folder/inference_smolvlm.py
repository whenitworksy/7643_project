import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login
import os
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge 
from bert_score import score as bert_score
from collections import defaultdict
import numpy as np

def load_existing_results(output_file: str) -> list[dict]:
    """Load results from existing output file"""
    results = []
    with open(output_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'image_name': row['image_name'],
                'image_path': row['image_path'],
                'ground_truth': row['ground_truth'],
                'generated_recipe': row['generated_recipe']
            })
    return results

def load_recipe_data(csv_path: str, image_folder: str) -> list[dict]:
    """Load recipe data from CSV and match with image paths"""
    recipe_data = []
    
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get image name from CSV
            image_name = row['Image_Name'] 
            image_path = os.path.join(image_folder, f"{image_name}.jpg")  # Adjust extension if needed
            
            if os.path.exists(image_path):
                recipe_data.append({
                    'image_path': image_path,
                    'image_name': image_name,
                    'ground_truth_recipe': row['Instructions'] 
                })
            else:
                print(f"Image not found: {image_path}")
    
    return recipe_data

def generate_text_from_image(batch_data, prompt_text="How should I cook this dish?"):
    # Load model and processor
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(
        model_id,
        size={"longest_edge": 512},
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # # Prepare multimodal input
    # image = Image.open(image_path).convert("RGB")
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": image},
    #             {"type": "text", "text": prompt_text}
    #         ]
    #     }
    # ]

    # Prepare all inputs
    images = []
    valid_data = []

    for item in batch_data:
        try:
            img = Image.open(item['image_path']).convert("RGB")
            images.append(img)
            valid_data.append(item)
        except Exception as e:
            print(f"Skipping {item['image_path']} due to error: {str(e)}")

    if not images:
        raise ValueError("No valid images in batch")

    # Create messages for each image-prompt pair
    batch_messages = []
    for image in images:
        batch_messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ])
    
    # Process all inputs together
    batch_texts = [processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) for messages in batch_messages]

    # Batch processing
    batch_inputs = processor(
        text=batch_texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    stop_token_ids = [processor.tokenizer.eos_token_id,
                      processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generation_params = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 1.0,
        "eos_token_id": stop_token_ids,
        "disable_compile": True
    }


    outputs = model.generate(**batch_inputs, **generation_params)
    
    # Trim input tokens from outputs and decode
    results = []
    for i, item in enumerate(valid_data):
        input_length = batch_inputs.input_ids[i].shape[0]
        generated_ids = outputs[i][input_length:]
        decoded_text = processor.decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        results.append({
            'image_name': item['image_name'],
            'image_path': item['image_path'],
            'ground_truth': item['ground_truth_recipe'],
            'generated_recipe': decoded_text
        })
    
    return results

def evaluate_recipes(ground_truth: str, generated: str) -> dict:
    """Calculate all metrics for a single recipe pair"""
    metrics = {}
    
    # Tokenize
    gt_tokens = ground_truth.split()
    gen_tokens = generated.split()
    
    # 1. BLEU (1-4 grams)
    smooth = SmoothingFunction().method1
    metrics['bleu'] = sentence_bleu(
        [gt_tokens], gen_tokens, 
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth
    )
    
    # # 2. ROUGE
    # rouge = Rouge()
    # scores = rouge.get_scores(generated, ground_truth)[0]
    # metrics.update({
    #     'rouge-l': scores['rouge-l']['f'],
    #     'rouge-1': scores['rouge-1']['f']
    # })
    
    # # 3. BERTScore
    # P, R, F1 = bert_score([generated], [ground_truth], lang='en')
    # metrics['bertscore'] = F1.mean().item()
    
    # # 4. Recipe-specific (simplified)
    # gt_ingredients = set([w for w in gt_tokens if w.endswith(',') or w.endswith('.')])
    # gen_ingredients = set([w for w in gen_tokens if w.endswith(',') or w.endswith('.')])
    
    # metrics.update({
    #     'ingredient_coverage': len(gt_ingredients & gen_ingredients) / max(1, len(gt_ingredients)),
    #     'hallucination_rate': len(gen_ingredients - gt_ingredients) / max(1, len(gen_ingredients))
    # })
    
    return metrics

def analyze_results(all_results: list) -> dict:
    """Aggregate metrics across all samples"""
    aggregated = defaultdict(list)
    
    for result in all_results:
        metrics = evaluate_recipes(result['ground_truth'], result['generated_recipe'])
        for k, v in metrics.items():
            aggregated[k].append(v)
    
    return {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        for metric, values in aggregated.items()
    }


# Usage
if __name__ == '__main__':
    HF_TOKEN = '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'
    print('Loginning in HF')
    login(HF_TOKEN)
    print('Done')

    # Path configuration
    csv_path = "../data/food_image_recipe_1k.csv"  # Update with your CSV path
    image_folder = "../data/food_images"
    output_file = "recipe_comparisons.csv"

    # Check for existing results
    if os.path.exists(output_file):
        print(f"Found existing results file: {output_file}")
        all_results = load_existing_results(output_file)
        print(f"Loaded {len(all_results)} existing results")
    else:
        # Load data and process if no existing results
        recipe_data = load_recipe_data(csv_path, image_folder)
        if not recipe_data:
            print("No valid image-recipe pairs found")
        print(f"Found {len(recipe_data)} valid image-recipe pairs")
        all_results = []

        # Process in batches
        batch_size = 4  
        all_results = []

        for i in range(0, len(recipe_data), batch_size):
            batch = recipe_data[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{len(recipe_data)//batch_size + 1}")
            
            try:
                batch_results = generate_text_from_image(batch)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        # Save results to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_name', 'image_path', 'ground_truth', 'generated_recipe']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)


    print(f"\nProcessing complete! Results saved to {output_file}")

    # After processing all batches:
    print("\nEvaluating results...")
    metrics_summary = analyze_results(all_results)

    # Print metrics
    print("\n=== Evaluation Summary ===")
    for metric, stats in metrics_summary.items():
        print(f"{metric.upper():<20}: {stats['mean']:.3f} ± {stats['std']:.3f}")
