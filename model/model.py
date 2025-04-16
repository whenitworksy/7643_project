import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import RecipeDecoder


class FoodImageToRecipeModel(nn.Module):
    """
    Complete model combining image encoder and recipe decoder
    """

    def __init__(self, embed_size=512, hidden_size=512, vocab_size=30000, num_layers=1):
        super(FoodImageToRecipeModel, self).__init__()

        self.encoder = ImageEncoder(embed_size)
        self.decoder = RecipeDecoder(embed_size, hidden_size, vocab_size, num_layers)
        self.vocab_size = vocab_size

    def forward(self, images, captions=None, teacher_forcing_ratio=1.0):
        """
        Forward pass
        If captions is provided, use teacher forcing for training
        If captions is None, sample a recipe sequence
        """
        # Extract features from images
        features = self.encoder(images)

        # If captions provided, use for training
        if captions is not None:
            return self.decoder(features, captions)
        else:
            # Sample a recipe during inference
            return self.decoder.sample(features), None