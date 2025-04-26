import json, random, shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
image_base_path = Path("data/train_data")
output_dir = Path("data/mini_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert image ID to nested file path
def image_id_to_path(image_id: str, base=image_base_path):
    return base / image_id[0] / image_id[1] / image_id[2] / image_id[3] / image_id  # includes .jpg

# Load JSONs
with open("data/recipe1m_images/layer1.json", "r", encoding="utf-8") as f:
    layer1 = {r["id"]: r for r in json.load(f)}

with open("data/recipe1m_images/layer2.json", "r", encoding="utf-8") as f:
    layer2 = json.load(f)

# Threading
def process_entry(entry):
    recipe_id = entry["id"]
    recipe = layer1.get(recipe_id)
    if not recipe or not recipe.get("ingredients") or not recipe.get("instructions"):
        return None

    for img in entry.get("images", []):
        img_id = img.get("id", "").lower().strip()
        img_path = image_id_to_path(img_id)
        if img_path.exists():
            return {
                "image": img_path.name,
                "image_id": img_id,
                "ingredients": [ing["text"].lower() for ing in recipe["ingredients"]],
                "instructions": [instr["text"] for instr in recipe["instructions"]],
            }
    return None

# Process entries in parallel
valid_recipes = []
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(process_entry, entry) for entry in layer2]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning training set"):
        result = future.result()
        if result:
            valid_recipes.append(result)

print(f"Valid matched training recipes: {len(valid_recipes)}")

# Sample and copy images
subset_size = min(10000, len(valid_recipes))
subset = random.sample(valid_recipes, subset_size)

for recipe in subset:
    src = image_id_to_path(recipe["image_id"])
    dst = output_dir / recipe["image"]
    shutil.copy(src, dst)

# Save JSON
with open(output_dir / "recipes.json", "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2)

print(f"Extracted {subset_size} training recipes to: {output_dir}")