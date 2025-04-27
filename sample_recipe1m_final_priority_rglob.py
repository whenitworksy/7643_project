import json, random, shutil
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# === SEED SETUP ===
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# === CONFIGURATION ===
LAYER1_PATH = Path("data/recipe1m_images/layer1.json")
LAYER2_PATH = Path("data/recipe1m_images/layer2.json")
DET_INGRS_PATH = Path("data/recipe1m_images/det_ingrs.json")
IMAGE_DIR = Path("data/train_data")
TRAIN_OUT = Path("data/mini_data")
VAL_OUT = Path("data/mini_data_val")
TRAIN_SIZE = 20000
VAL_SIZE = 3000
STRATIFY_BY = "ingredient_category"
MAX_PER_STRATUM = 10000

# === MAPPING SETUP ===
INGREDIENT_CATEGORY_MAP = {
    "salt": "spice", "kosher salt": "spice", "salt and pepper": "spice", "black pepper": "spice", "pepper": "spice",
    "fresh ground black pepper": "spice", "ground cinnamon": "spice", "cinnamon": "spice",
    "garlic cloves": "spice", "garlic clove": "spice", "garlic": "spice", "garlic powder": "spice",
    "butter": "dairy", "unsalted butter": "dairy", "milk": "dairy", "sour cream": "dairy", "cream cheese": "dairy", "heavy cream": "dairy", "parmesan cheese": "dairy",
    "sugar": "sweetener", "brown sugar": "sweetener", "granulated sugar": "sweetener", "honey": "sweetener", "vanilla": "sweetener", "vanilla extract": "sweetener",
    "olive oil": "oil", "extra virgin olive oil": "oil", "vegetable oil": "oil", "oil": "oil",
    "onion": "vegetable", "onions": "vegetable", "green onions": "vegetable", "tomatoes": "vegetable", "carrots": "vegetable", "celery": "vegetable",
    "flour": "baking", "all-purpose flour": "baking", "baking powder": "baking", "baking soda": "baking", "cornstarch": "baking",
    "water": "liquid", "lemon juice": "liquid", "fresh lemon juice": "liquid", "chicken broth": "liquid",
    "soy sauce": "condiment", "worcestershire sauce": "condiment", "mayonnaise": "condiment",
}

CATEGORY_PRIORITY = {
    "meat": 1, "seafood": 1, "dairy": 2, "grain": 3, "baking": 3, "vegetable": 4, "sweetener": 4,
    "oil": 5, "spice": 5, "liquid": 5, "condiment": 5, "unknown": 6,
}

skip_stats = defaultdict(int)

# === FUNCTION DEFINITIONS ===

def load_jsons():
    with open(LAYER1_PATH, "r", encoding="utf-8") as f:
        layer1 = {r["id"]: r for r in json.load(f)}
    with open(LAYER2_PATH, "r", encoding="utf-8") as f:
        layer2 = json.load(f)
    with open(DET_INGRS_PATH, "r", encoding="utf-8") as f:
        det_data = json.load(f)
    det_map = {}
    for entry in det_data:
        recipe_id = entry["id"]
        valid_flags = entry["valid"]
        ingredients = entry["ingredients"]
        filtered_ingredients = [
            ing["text"].lower().strip()
            for ing, is_valid in zip(ingredients, valid_flags)
            if is_valid and "text" in ing
        ]
        det_map[recipe_id] = filtered_ingredients
    return layer1, layer2, det_map

def normalize_ingredient_name(name):
    name = name.lower().strip()
    if name.endswith("es") and name[:-2] in INGREDIENT_CATEGORY_MAP:
        return name[:-2]
    if name.endswith("s") and name[:-1] in INGREDIENT_CATEGORY_MAP:
        return name[:-1]
    return name

def map_ingredient_to_category(ingredient):
    normalized = normalize_ingredient_name(ingredient)
    return INGREDIENT_CATEGORY_MAP.get(normalized, "unknown")

def pick_best_category(categories):
    if not categories:
        return "unknown"
    return min(categories, key=lambda cat: CATEGORY_PRIORITY.get(cat, 6))

def process_entry_for_sampling(entry, layer1, det_map):
    recipe_id = entry["id"]
    recipe = layer1.get(recipe_id)
    if not recipe:
        return None, "no_layer1_entry"
    if len(recipe.get("ingredients", [])) < 2 or len(recipe.get("instructions", [])) < 2:  # 🔥 relaxed rule!
        return None, "too_few_ingredients_or_instructions"
    if len(recipe.get("instructions", [])) > 50:
        return None, "too_many_instructions"
    if not entry.get("images"):
        return None, "no_images"
    img_data = random.choice(entry["images"])
    img_id = img_data.get("id", "").lower().strip()
    if not img_id:
        return None, "no_valid_image_id"
    detected_ingredients = det_map.get(recipe_id, [])
    categories = [map_ingredient_to_category(i) for i in detected_ingredients if map_ingredient_to_category(i) != "unknown"]
    return {
        "recipe_id": recipe_id,
        "image_id": img_id,
        "image": img_id,
        "ingredients": [i["text"].lower() for i in recipe["ingredients"]],
        "instructions": [i["text"] for i in recipe["instructions"]],
        "ingredient_category": pick_best_category(categories) if categories else "unknown",
    }, None

def create_stratified_lists(all_recipes, stratify_by, train_size, val_size, max_per_stratum):
    grouped_recipes = defaultdict(list)
    for recipe in tqdm(all_recipes, desc="Grouping by category"):
        grouped_recipes[recipe[stratify_by]].append(recipe)
    train_list, val_list = [], []
    for stratum, recipes in grouped_recipes.items():
        random.shuffle(recipes)
        num_in_stratum = len(recipes)
        train_split = int(train_size * (num_in_stratum / len(all_recipes)))
        val_split = int(val_size * (num_in_stratum / len(all_recipes)))
        train_take = min(train_split, num_in_stratum, max_per_stratum)
        val_take = min(val_split, num_in_stratum - train_take, max_per_stratum // 3)
        train_list.extend(recipes[:train_take])
        val_list.extend(recipes[train_take : train_take + val_take])
    random.shuffle(train_list)
    random.shuffle(val_list)
    return train_list, val_list

def backfill_if_needed(list_to_fill, all_candidates, target_size, seen_ids_set):
    needed = target_size - len(list_to_fill)
    if needed <= 0:
        return list_to_fill
    candidates = [r for r in all_candidates if r["recipe_id"] not in seen_ids_set]
    random.shuffle(candidates)
    list_to_fill.extend(candidates[:needed])
    return list_to_fill

def save_lists(train_list, val_list, train_out_dir, val_out_dir):
    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir.mkdir(parents=True, exist_ok=True)
    with open(train_out_dir / "train_recipes_stratified.json", "w", encoding="utf-8") as f:
        json.dump(train_list, f, indent=2)
    with open(val_out_dir / "val_recipes_stratified.json", "w", encoding="utf-8") as f:
        json.dump(val_list, f, indent=2)

def copy_images_fast(recipe_list, image_source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    print("Scanning available images...")
    image_lookup = {}
    for img_path in tqdm(image_source_dir.rglob("*.jpg"), desc="Building image lookup"):
        key = img_path.name.lower().strip()
        image_lookup[key] = img_path

    for recipe in tqdm(recipe_list, desc=f"Copying images to {output_dir}"):
        img_filename = recipe["image"].lower().strip()
        src_path = image_lookup.get(img_filename, None)
        if src_path:
            dst_path = output_dir / img_filename
            shutil.copy(src_path, dst_path)
            copied += 1
        else:
            skip_stats["missing_image_file"] += 1
    print(f"Copied {copied}/{len(recipe_list)} images to {output_dir}")

def filter_existing_images(recipe_list, image_dir):
    new_list = []
    for recipe in tqdm(recipe_list, desc="Filtering recipes with existing images"):
        img_path = image_dir / recipe["image"]
        if img_path.exists():
            new_list.append(recipe)
    return new_list

def plot_distribution(recipe_list, title="Ingredient Category Distribution"):
    counter = Counter([r["ingredient_category"] for r in recipe_list])
    plt.bar(*zip(*counter.items()))
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def generate_summary_tables(train_list, val_list, train_dir, val_dir):
    def summarize(recipes, name):
        category_counter = Counter([r["ingredient_category"] for r in recipes])
        total = len(recipes)
        rows = []
        for cat, count in category_counter.most_common():
            rows.append([cat, count, f"{100 * count / total:.1f}%", ", ".join([k for k, v in INGREDIENT_CATEGORY_MAP.items() if v == cat][:3])])
        print(f"\n{name} Category Breakdown:")
        print(tabulate(rows, headers=["Category", "# Recipes", "% of Set", "Example Ingredients"]))
        return len(category_counter)

    print("\n--- Skipped Recipe Summary ---")
    for reason, count in skip_stats.items():
        print(f"{reason:<35}: {count}")

    train_cat_count = summarize(train_list, "Train")
    val_cat_count = summarize(val_list, "Validation")
    summary_rows = [
        ["Train", len(train_list), train_cat_count, len(list(train_dir.glob('*.jpg'))), str(train_dir / "train_recipes_stratified.json")],
        ["Validation", len(val_list), val_cat_count, len(list(val_dir.glob('*.jpg'))), str(val_dir / "val_recipes_stratified.json")],
    ]
    print("\nDataset Overview:")
    print(tabulate(summary_rows, headers=["Split", "Total Recipes", "Unique Categories", "Images Copied", "JSON File Location"]))

def generate_maximum_recipe_summary(layer1, layer2, det_map, image_dir):
    valid_recipes = []
    missing_img = 0

    def get_actual_image_path(image_dir, img_filename):
        stem = Path(img_filename).stem.lower()
        return image_dir / stem[0] / stem[1] / stem[2] / stem[3] / img_filename

    for entry in tqdm(layer2, desc="Scanning all recipes for maximum usable subset"):
        processed, reason = process_entry_for_sampling(entry, layer1, det_map)
        if processed:
            img_filename = processed["image"].lower().strip()
            img_path = get_actual_image_path(image_dir, img_filename)
            if img_path.exists():
                valid_recipes.append(processed)
            else:
                missing_img += 1

    print(f"\n✅ Total usable recipes with matching images: {len(valid_recipes)}")
    print(f"❌ Total skipped due to missing image files: {missing_img}")

    counter = Counter([r["ingredient_category"] for r in valid_recipes])
    total = sum(counter.values())
    rows = []
    for cat, count in counter.most_common():
        rows.append([cat, count, f"{100 * count / total:.2f}%"])

    print("\n📊 Maximum Recipes per Category (Based on Existing Images):")
    print(tabulate(rows, headers=["Category", "# Recipes", "% of Total"]))

    return valid_recipes

# === MAIN EXECUTION ===
# layer1, layer2, det_map = load_jsons()
# all_processed = []
# seen_ids = set()
# for entry in tqdm(layer2, desc="Processing recipes"):
#     processed, reason = process_entry_for_sampling(entry, layer1, det_map)
#     if processed:
#         if processed["recipe_id"] not in seen_ids:
#             all_processed.append(processed)
#             seen_ids.add(processed["recipe_id"])
#         else:
#             skip_stats["duplicate_recipe_id"] += 1
#     else:
#         if reason:
#             skip_stats[reason] += 1

# train_list, val_list = create_stratified_lists(all_processed, STRATIFY_BY, TRAIN_SIZE, VAL_SIZE, MAX_PER_STRATUM)
# for r in train_list + val_list:
#     seen_ids.add(r["recipe_id"])

# train_list = backfill_if_needed(train_list, all_processed, TRAIN_SIZE, seen_ids)
# val_list = backfill_if_needed(val_list, all_processed, VAL_SIZE, seen_ids)

# print(f"Train size before copy: {len(train_list)}")
# print(f"Val size before copy: {len(val_list)}")

# copy_images_fast(train_list, IMAGE_DIR, TRAIN_OUT)
# copy_images_fast(val_list, IMAGE_DIR, VAL_OUT)

# # Filter to match only existing images
# train_list = filter_existing_images(train_list, TRAIN_OUT)
# val_list = filter_existing_images(val_list, VAL_OUT)

# # Save clean JSONs again
# save_lists(train_list, val_list, TRAIN_OUT, VAL_OUT)

# plot_distribution(train_list, "Train Set Ingredient Category Distribution")
# generate_summary_tables(train_list, val_list, TRAIN_OUT, VAL_OUT)

layer1, layer2, det_map = load_jsons()
valid_recipes = generate_maximum_recipe_summary(layer1, layer2, det_map, IMAGE_DIR)