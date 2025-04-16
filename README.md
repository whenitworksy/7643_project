'''
# Food Image to Recipe Generator

This project uses deep learning to generate cooking recipes from food images. It employs a CNN-LSTM architecture with an image encoder (ResNet50) and recipe text decoder.

## Project Structure

```
/food_recipe_generator/
├── main.py                  # Main script for training/inference
├── model/                   # Model architecture
├── data/                    # Dataset and preprocessing
├── utils/                   # Utilities for checkpointing and logging
└── notebooks/               # Demo notebooks
```

## Setup

1. Clone the repository
2. Install requirements:
   ```
   pip install torch torchvision pillow matplotlib tqdm
   ```
3. Prepare your dataset (Recipe1M+ recommended)

## Usage

### Training

Train the model using food images and recipes:

```bash
python main.py train --data-dir /path/to/recipe_dataset --batch-size 32 --num-epochs 50
```

Key training arguments:
- `--data-dir`: Path to dataset
- `--batch-size`: Batch size for training
- `--num-epochs`: Number of training epochs
- `--learning-rate`: Learning rate
- `--output-dir`: Directory to save model checkpoints and logs

### Generating Recipes

Generate a recipe from a food image:

```bash
python main.py generate --image-path /path/to/food.jpg --output-recipe recipe.txt
```

Key generation arguments:
- `--image-path`: Path to input food image
- `--model-path`: Path to model checkpoint (optional)
- `--output-recipe`: Path to save generated recipe

## Checkpointing

The model automatically saves checkpoints during training:
- Regular checkpoints every epoch
- Best model based on validation loss
- Training can be resumed from the latest checkpoint

## Dataset Format

The expected dataset format follows Recipe1M+ structure:
- JSON files with recipe information
- Image paths linked to recipes
- Each recipe includes title, ingredients, and instructions

## Demo Notebook

Try the demo notebook in `notebooks/food_recipe_demo.ipynb` to:
- Generate recipes from sample images
- Upload your own food images
- Process multiple images in batch
- View training statistics

## References

- Recipe1M+: Building, Mining, and Exploring a Large-Scale Recipe Dataset
- Im2Recipe: Learning Cross-modal Embeddings for Cooking Recipes and Food Images
'''
