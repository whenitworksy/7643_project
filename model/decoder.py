import torch
import torch.nn as nn
import torch.nn.functional as F


class RecipeDecoder(nn.Module):
    """
    LSTM decoder for generating recipes
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RecipeDecoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        self.embed.weight.data.normal_(0.0, 0.02)
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, features, captions, hidden=None):
        """
        Forward pass for training with teacher forcing
        """
        embeddings = self.embed(captions)

        # If provided, concatenate the image features to the embeddings
        if features is not None:
            # Expand features to match sequence dimension of embeddings
            features = features.unsqueeze(1)

            # Use features as the first token's embedding
            embeddings = torch.cat((features, embeddings[:, 1:, :]), dim=1)

        # Pass through LSTM
        embeddings = self.dropout(embeddings)
        hiddens, states = self.lstm(embeddings, hidden)
        outputs = self.fc(hiddens)

        return outputs, states

    def sample(self, features, max_len=100, start_token=1, end_token=2):
        """
        Sample a recipe during inference (no teacher forcing)
        """
        batch_size = features.size(0)
        hidden = None

        # Start with start token for each sample in batch
        inputs = torch.LongTensor([start_token] * batch_size).to(features.device)
        sampled_ids = []

        for i in range(max_len):
            # Forward pass
            inputs_embed = self.embed(inputs).unsqueeze(1)

            # For first step, use image features to initialize hidden state
            if i == 0 and hidden is None:
                inputs_embed = features.unsqueeze(1)

            # LSTM step
            output, hidden = self.lstm(inputs_embed, hidden)
            output = self.fc(output.squeeze(1))

            # Sample next token
            _, predicted = output.max(1)
            sampled_ids.append(predicted)

            # Update inputs for next step
            inputs = predicted

            # If all samples hit the end token, stop early
            if (predicted == end_token).all():
                break

        # Stack sampled tokens
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids