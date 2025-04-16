import os
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
from PIL import Image
import numpy as np


class RecipeDataset(Dataset):
    """
    Dataset for food image to recipe generation
    """

    def __init__(self, data_root, preprocessor, split='train', max_samples=None):
        self.data_root = data_root
        self.preprocessor = preprocessor
        self.split = split

        # Load data
        print(f"Loading {split} dataset...")
        self.data = self.load_data(max_samples)
        print(f"Loaded {len(self.data)} samples for {split}")

    def load_data(self, max_samples=None):
        """
        Load data entries from Recipe1M+ dataset
        Expected format similar to Recipe1M structure
        """
        data = []

        # Path to dataset index file
        index_path = os.path.join(self.data_root, f'{self.split}.json')

        if not os.path.exists(index_path):
            print(f"Warning: {index_path} not found. Using placeholder data.")
            # Return placeholder data for development
            return [{'image_path': 'placeholder.jpg',
                     'recipe_text': 'placeholder recipe'} for _ in range(10)]

        try:
            # Load dataset index
            with open(index_path, 'r') as f:
                dataset_index = json.load(f)

            # Process entries
            for item in dataset_index:
                # Extract paths and texts based on Recipe1M+ structure
                # This may need to be adapted to the actual dataset structure
                image_path = os.path.join(self.data_root, 'images', item['image_path'])

                # Combine recipe title, ingredients and instructions into text
                recipe_text = item['title'] + '. '
                recipe_text += 'Ingredients: ' + ' '.join(item['ingredients']) + '. '
                recipe_text += 'Instructions: ' + ' '.join(item['instructions'])

                data.append({
                    'image_path': image_path,
                    'recipe_text': recipe_text
                })

                # Limit samples if specified
                if max_samples and len(data) >= max_samples:
                    break

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using placeholder data instead.")
            return [{'image_path': 'placeholder.jpg',
                     'recipe_text': 'placeholder recipe'} for _ in range(10)]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and preprocess image
        image = self.preprocessor.preprocess_image(item['image_path'])

        # Tokenize recipe
        recipe_tokens = self.preprocessor.tokenize_recipe(item['recipe_text'])

        return image, recipe_tokens