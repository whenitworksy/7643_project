import json, random, shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
image_base_path = Path("data/train_data")
output_dir = Path("data/mini_data_val")
output_dir.mkdir(parents=True, exist_ok=True)
train_image_ids_file = Path("data/mini_data/recipes.json")

# === Load image IDs from training subset to avoid overlap ===
with open(train_image_ids_file, "r", encoding="utf-8") as f:
    train_image_ids = set(r["image_id"] for r in json.load(f))

# === Convert image ID to nested file path ===
def image_id_to_path(image_id: str, base=image_base_path):
    return base / image_id[0] / image_id[1] / image_id[2] / image_id[3] / image_id  # includes .jpg

# === Load JSONs ===
with open("data/recipe1m_images/layer1.json", "r", encoding="utf-8") as f:
    layer1 = {r["id"]: r for r in json.load(f)}

with open("data/recipe1m_images/layer2.json", "r", encoding="utf-8") as f:
    layer2 = json.load(f)

# === Worker function for threading ===
def process_entry(entry):
    recipe_id = entry["id"]
    recipe = layer1.get(recipe_id)
    if not recipe or not recipe.get("ingredients") or not recipe.get("instructions"):
        return None

    for img in entry.get("images", []):
        img_id = img.get("id", "").lower().strip()
        if img_id in train_image_ids:
            continue
        img_path = image_id_to_path(img_id)
        if img_path.exists():
            return {
                "image": img_path.name,
                "image_id": img_id,
                "ingredients": [ing["text"].lower() for ing in recipe["ingredients"]],
                "instructions": [instr["text"] for instr in recipe["instructions"]],
            }
    return None

# === Process entries in parallel ===
valid_recipes = []
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(process_entry, entry) for entry in layer2]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning layer2 for validation set"):
        result = future.result()
        if result:
            valid_recipes.append(result)

print(f"Valid validation recipes (non-overlapping): {len(valid_recipes)}")

# === Sample and copy images ===
subset_size = min(1000, len(valid_recipes))
subset = random.sample(valid_recipes, subset_size)

for recipe in subset:
    src = image_id_to_path(recipe["image_id"])
    dst = output_dir / recipe["image"]
    shutil.copy(src, dst)

# === Save JSON ===
with open(output_dir / "recipes.json", "w", encoding="utf-8") as f:
    json.dump(subset, f, indent=2)

print(f"Extracted validation set of {subset_size} recipes to: {output_dir}")