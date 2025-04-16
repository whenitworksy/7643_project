import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
import json
from collections import Counter


class RecipeDataPreprocessor:
    """
    Handles preprocessing of images and text for recipe generation
    """

    def __init__(self, image_size=224, max_vocab_size=30000, vocab_threshold=5):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # For text preprocessing
        self.vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.inv_vocab = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.vocab_size = 4  # Start with special tokens
        self.max_vocab_size = max_vocab_size
        self.vocab_threshold = vocab_threshold

    def build_vocab(self, recipe_texts, save_path=None):
        """
        Build vocabulary from recipe texts
        """
        print("Building vocabulary...")
        word_counts = Counter()

        # Count words
        for recipe in recipe_texts:
            word_counts.update(recipe.split())

        # Filter words by frequency threshold
        words = [word for word, count in word_counts.items()
                 if count >= self.vocab_threshold]

        # Sort by frequency
        words = sorted(words, key=lambda x: word_counts[x], reverse=True)

        # Limit vocabulary size
        if len(words) > self.max_vocab_size - 4:  # -4 for special tokens
            words = words[:self.max_vocab_size - 4]

        # Build vocabulary
        for word in words:
            self.vocab[word] = self.vocab_size
            self.inv_vocab[self.vocab_size] = word
            self.vocab_size += 1

        print(f"Vocabulary built with {self.vocab_size} words")

        # Save vocabulary
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'inv_vocab': self.inv_vocab,
                    'vocab_size': self.vocab_size
                }, f)
            print(f"Vocabulary saved to {save_path}")

    def load_vocab(self, path):
        """
        Load vocabulary from file
        """
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
            self.vocab = vocab_data['vocab']
            self.inv_vocab = vocab_data['inv_vocab']
            self.vocab_size = vocab_data['vocab_size']
        print(f"Loaded vocabulary with {self.vocab_size} words")

    def tokenize_recipe(self, recipe_text, max_length=300):
        """
        Convert recipe text to token indices
        """
        words = recipe_text.split()
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]

        # Add start and end tokens
        tokens = [self.vocab["<START>"]] + tokens + [self.vocab["<END>"]]

        # Pad or truncate to fixed length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens += [self.vocab["<PAD>"]] * (max_length - len(tokens))

        return torch.tensor(tokens)

    def decode_recipe(self, token_indices):
        """
        Convert token indices back to recipe text
        """
        recipe_words = []
        for idx in token_indices:
            word = self.inv_vocab.get(idx.item(), "<UNK>")
            if word == "<END>":
                break
            if word not in ["<PAD>", "<START>"]:
                recipe_words.append(word)
        return " ".join(recipe_words)

    def preprocess_image(self, image_path):
        """
        Load and preprocess an image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, 224, 224)