import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import get_model_weights
from transformers import AutoTokenizer, AutoModel

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512, model_name='resnet18', tune_last_n_blocks=0, dropout=0.1):
        super().__init__()
        weights_enum = get_model_weights(model_name)
        weights = weights_enum.DEFAULT
        backbone = models.__dict__[model_name](weights=weights)

        self.model_name = model_name
        self.tune_last_n_blocks = tune_last_n_blocks

        if model_name.startswith("resnet"):
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.proj_in_dim = backbone.fc.in_features
            if tune_last_n_blocks == 0:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                blocks = list(self.backbone.children())
                for i, block in enumerate(blocks):
                    requires_grad = i >= len(blocks) - tune_last_n_blocks
                    for param in block.parameters():
                        param.requires_grad = requires_grad

        elif model_name.startswith("efficientnet"):
            self.backbone = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj_in_dim = backbone.classifier[1].in_features
            for i, block in enumerate(self.backbone):
                requires_grad = i >= len(self.backbone) - tune_last_n_blocks
                for param in block.parameters():
                    param.requires_grad = requires_grad

        elif model_name.startswith("vit"):
            self.backbone = backbone
            self.proj_in_dim = backbone.heads.head.in_features
            encoder_layers = self.backbone.encoder.layers
            for i, layer in enumerate(encoder_layers):
                requires_grad = i >= len(encoder_layers) - tune_last_n_blocks
                for param in layer.parameters():
                    param.requires_grad = requires_grad

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.fc = nn.Sequential(  # Add dropout to linear layer in sequence
            nn.Linear(self.proj_in_dim, embed_dim),
            nn.Dropout(dropout)
        )
        nn.init.xavier_uniform_(self.fc[0].weight)  # image projection layer

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        if len(x.shape) == 4:
            x = x.squeeze()
        x = self.fc(x)
        return F.normalize(x, dim=-1)
    
    def print_trainable_layers(self):
        print(f"\nTrainable layers in {self.model_name}:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}")

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=2, num_heads=4, max_len=128, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout) 
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids):
        x = self.token_embed(token_ids)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)  # Dropout on input tokens
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.out_proj(x), dim=-1)

class BertIngredientEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embed_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def tokenize_ingredients(self, ingredient_list, max_len=64):
        text = ", ".join(ingredient_list)
        return self.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

    def forward(self, ingredient_batch):
        batch = [", ".join(ings) for ings in ingredient_batch]
        encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.cuda() for k, v in encoded.items()}
        outputs = self.bert(**encoded).last_hidden_state
        pooled = self.pool(outputs.transpose(1, 2)).squeeze(-1)
        return F.normalize(self.project(pooled), dim=-1)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)  # Dropout in attention
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout after ReLU
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)   # Dropout after final FFN layer
        )
        nn.init.xavier_uniform_(self.ff[0].weight)  # for FFN first layer
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        attn_output, _ = self.cross_attn(query, context, context)
        x = self.norm(query + self.dropout1(attn_output))  # Dropout on residual connection
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)

class JointEncoderWithCrossAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_dropout = nn.Dropout(dropout)  
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, img_emb, text_tokens):
        img_seq = self.image_proj(img_emb).unsqueeze(1)
        txt_emb = self.text_dropout(self.text_embed(text_tokens).transpose(1, 2))  # Add dropout here
        txt_pooled = self.text_pool(txt_emb).squeeze(-1).unsqueeze(1)
        x = img_seq
        for layer in self.cross_layers:
            x = layer(x, txt_pooled)
        return F.normalize(self.final_proj(x.squeeze(1)), dim=-1)

class MultiModalEncoder(nn.Module):
    def __init__(self, mode='dual', vocab_size=10000, embed_dim=512, tune_last_n_blocks=0, dropout=0.1):
        super().__init__()
        self.mode = mode
        self.img_encoder = ImageEncoder(embed_dim, tune_last_n_blocks=tune_last_n_blocks, dropout=dropout)

        if mode == 'dual':
            self.txt_encoder = TransformerEncoder(vocab_size, embed_dim, dropout=dropout)
        elif mode == 'fused':
            self.ingr_encoder = TransformerEncoder(vocab_size, embed_dim, dropout=dropout)
            self.instr_encoder = TransformerEncoder(vocab_size, embed_dim, dropout=dropout)
            self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        elif mode == 'joint':
            self.cross_encoder = JointEncoderWithCrossAttention(vocab_size, embed_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, img, tokens1, tokens2=None):
        img_emb = self.img_encoder(img)
        if self.mode == 'dual':
            txt_emb = self.txt_encoder(tokens1)
        elif self.mode == 'fused':
            ingr_emb = self.ingr_encoder(tokens1)
            instr_emb = self.instr_encoder(tokens2)
            fused = torch.cat([ingr_emb, instr_emb], dim=-1)
            txt_emb = F.normalize(self.fusion(fused), dim=-1)
        elif self.mode == 'joint':
            txt_emb = self.cross_encoder(img_emb, tokens1)
        return img_emb, txt_emb