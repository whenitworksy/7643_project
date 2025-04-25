import json
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
MODEL_NAME = "google/gemma-2b"
RETRIEVAL_PATH = "retrieval_results.json"
OUT_PATH = "generated_topk_batched.json"
TOPK = 5
BATCH_SIZE = 4
MAX_NEW_TOKENS = 40
MAX_PROMPT_LEN = 256
MAX_INGREDIENTS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
model = model.to(DEVICE).eval()

# Load retrievals
with open(RETRIEVAL_PATH, "r") as f:
    retrievals = json.load(f)

# Generate in batched mode
results = []
batch_prompts = []
batch_meta = []
start_time = time.time()

for item in tqdm(retrievals, desc="Preparing Prompts"):
    query_image = item["query_image"]
    for i, recipe in enumerate(item["top_recipes"][:TOPK]):
        trimmed_ingredients = recipe["ingredients"][:MAX_INGREDIENTS]
        prompt = f"Ingredients: {', '.join(trimmed_ingredients)}\nInstructions:"
        batch_prompts.append(prompt)
        batch_meta.append({"query_image": query_image, "rank": i + 1, "ingredients": trimmed_ingredients})

# Batched inference
all_generations = []
for i in tqdm(range(0, len(batch_prompts), BATCH_SIZE), desc="Generating Batches"):
    batch = batch_prompts[i:i + BATCH_SIZE]
    meta = batch_meta[i:i + BATCH_SIZE]

    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_LEN).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for meta_item, gen_text in zip(meta, decoded):
        all_generations.append({
            **meta_item,
            "generated_instructions": gen_text
        })

# Group back into per-image format
from collections import defaultdict
image_dict = defaultdict(list)
for entry in all_generations:
    image_dict[entry["query_image"]].append({
        "rank": entry["rank"],
        "ingredients": entry["ingredients"],
        "generated_instructions": entry["generated_instructions"]
    })

final_results = [
    {"query_image": k, "topk_generations": sorted(v, key=lambda x: x["rank"])}
    for k, v in image_dict.items()
]

# Save
with open(OUT_PATH, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"Saved {len(final_results)} batched generations to {OUT_PATH}")
print(f"Total time: {time.time() - start_time:.2f}s")