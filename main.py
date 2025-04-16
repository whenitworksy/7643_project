import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import time

# Import project modules
from model.model import FoodImageToRecipeModel
from data.dataset import RecipeDataset
from data.preprocessor import RecipeDataPreprocessor
from utils.checkpointing import CheckpointManager
from utils.logger import TrainingLogger


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def validate(model, val_loader, criterion, device, preprocessor, num_examples=2):
    """
    Validate the model on validation set
    Returns validation loss and example predictions
    """
    model.eval()
    val_loss = 0
    examples = []

    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs, _ = model(images, captions)

            # Calculate loss (ignoring padding tokens)
            targets = captions[:, 1:]  # Shift to get target tokens (start from second token)
            outputs = outputs[:, :-1, :]  # Remove last prediction

            # Reshape for cross entropy loss
            loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))
            val_loss += loss.item()

            # Generate example predictions for visualization
            if i == 0 and len(examples) < num_examples:
                # Generate predictions
                sampled_ids, _ = model(images[:num_examples])

                for j in range(min(num_examples, len(images))):
                    # For simplicity, use placeholder image path
                    img_path = f"val_image_{j}"

                    # Get target recipe (remove padding, start and end tokens)
                    target_recipe = preprocessor.decode_recipe(captions[j].cpu())

                    # Get predicted recipe
                    pred_recipe = preprocessor.decode_recipe(sampled_ids[j].cpu())

                    examples.append((img_path, target_recipe, pred_recipe))

    return val_loss / len(val_loader), examples


def train(args):
    """
    Train the model
    """
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Initialize data preprocessor
    preprocessor = RecipeDataPreprocessor(
        image_size=args.image_size,
        max_vocab_size=args.vocab_size,
        vocab_threshold=args.vocab_threshold
    )

    # Check if vocabulary exists, otherwise build it
    vocab_path = os.path.join(args.output_dir, 'vocabulary.pkl')
    if os.path.exists(vocab_path) and not args.rebuild_vocab:
        print(f"Loading vocabulary from {vocab_path}")
        preprocessor.load_vocab(vocab_path)
    else:
        print("Building vocabulary...")
        # This is a placeholder - you'll need to implement this based on your dataset
        # For example, load all recipe texts and build vocabulary

        # Load train dataset for building vocabulary
        temp_dataset = RecipeDataset(
            args.data_dir,
            preprocessor,
            split='train',
            max_samples=args.vocab_max_samples
        )

        # Extract all recipe texts
        recipe_texts = [item['recipe_text'] for item in temp_dataset.data]

        # Build and save vocabulary
        preprocessor.build_vocab(recipe_texts, save_path=vocab_path)

    # Create datasets
    train_dataset = RecipeDataset(
        args.data_dir,
        preprocessor,
        split='train',
        max_samples=args.max_samples
    )

    val_dataset = RecipeDataset(
        args.data_dir,
        preprocessor,
        split='val',
        max_samples=args.max_samples_val
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    model = FoodImageToRecipeModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=preprocessor.vocab_size,
        num_layers=args.num_layers
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=preprocessor.vocab["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.lr_patience,
        verbose=True
    )

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        model,
        optimizer,
        scheduler,
        keep_n_latest=args.keep_n_checkpoints
    )

    # Initialize logger
    logger = TrainingLogger(log_dir, model_name="food2recipe")

    # Resume from checkpoint if available
    start_epoch, _ = checkpoint_manager.load_latest_checkpoint()
    start_epoch = start_epoch + 1  # Start from next epoch

    # Training loop
    best_val_loss = float('inf')
    print(f"Starting training from epoch {start_epoch}/{args.num_epochs}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_loss = 0
        start_time = time.time()
        batch_count = 0
        total_samples = 0

        # Progress bar for batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")

        for images, captions in progress_bar:
            batch_count += 1
            total_samples += images.size(0)

            # Move to device
            images = images.to(device)
            captions = captions.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(images, captions)

            # Calculate loss (ignoring padding tokens)
            targets = captions[:, 1:]  # Shift to get target tokens
            outputs = outputs[:, :-1, :]  # Remove last prediction

            # Reshape for cross entropy loss
            loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))

            # Backward pass and optimize
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Update running loss
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{train_loss / batch_count:.4f}"
            })

            # Free up memory
            del images, captions, outputs, loss

        # Calculate average loss
        train_loss /= batch_count

        # Calculate training speed
        epoch_time = time.time() - start_time
        samples_per_sec = total_samples / epoch_time

        # Validate model
        val_loss, examples = validate(model, val_loader, criterion, device, preprocessor)

        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_manager.save_checkpoint(
            epoch,
            val_loss=val_loss,
            is_best=is_best,
            extra_data={
                'train_loss': train_loss,
                'samples_per_sec': samples_per_sec
            }
        )

        # Log progress
        logger.log_epoch(
            epoch,
            train_loss,
            val_loss=val_loss,
            lr=current_lr,
            samples_per_sec=samples_per_sec,
            examples=examples
        )

        # Plot and save training history
        if epoch % args.plot_every == 0 or epoch == args.num_epochs - 1:
            history_plot_path = os.path.join(log_dir, f"training_history_epoch_{epoch}.png")
            logger.plot_history(save_path=history_plot_path)

    print("Training completed!")

    # Load best model for final evaluation
    best_checkpoint = checkpoint_manager.load_best_checkpoint()
    if best_checkpoint:
        print(
            f"Loaded best model from epoch {best_checkpoint['epoch']} with validation loss {best_checkpoint['val_loss']:.4f}")

    # Final validation
    final_val_loss, _ = validate(model, val_loader, criterion, device, preprocessor, num_examples=5)
    print(f"Final validation loss: {final_val_loss:.4f}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': preprocessor.vocab_size,
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")


def generate(args):
    """
    Generate recipe from an image
    """
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize data preprocessor
    preprocessor = RecipeDataPreprocessor(image_size=args.image_size)

    # Load vocabulary
    vocab_path = os.path.join(args.output_dir, 'vocabulary.pkl')
    if not os.path.exists(vocab_path):
        print(f"Vocabulary file not found at {vocab_path}")
        return
    preprocessor.load_vocab(vocab_path)

    # Load model configuration
    if args.model_path and os.path.exists(args.model_path):
        checkpoint_path = args.model_path
    else:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.output_dir, 'final_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model parameters
    embed_size = checkpoint.get('embed_size', args.embed_size)
    hidden_size = checkpoint.get('hidden_size', args.hidden_size)
    num_layers = checkpoint.get('num_layers', args.num_layers)

    # Initialize model
    model = FoodImageToRecipeModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=preprocessor.vocab_size,
        num_layers=num_layers
    ).to(device)

    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Model loaded from {checkpoint_path}")

    # Set model to evaluation mode
    model.eval()

    # Load and preprocess image
    if not os.path.exists(args.image_path):
        print(f"Image not found at {args.image_path}")
        return

    image = preprocessor.preprocess_image(args.image_path)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Generate recipe
    with torch.no_grad():
        sampled_ids, _ = model(image)

    # Convert indices to words
    recipe = preprocessor.decode_recipe(sampled_ids[0].cpu())

    print("\nGenerated Recipe:")
    print("-" * 40)
    print(recipe)
    print("-" * 40)

    # Save recipe to file if output path provided
    if args.output_recipe:
        with open(args.output_recipe, 'w') as f:
            f.write(recipe)
        print(f"Recipe saved to {args.output_recipe}")


def main():
    parser = argparse.ArgumentParser(description='Food Image to Recipe Generator')
    subparsers = parser.add_subparsers(dest='mode', help='train or generate')

    # Common arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--embed-size', type=int, default=512, help='Embedding size')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')

    # Training arguments
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--lr-patience', type=int, default=5, help='LR scheduler patience')
    train_parser.add_argument('--vocab-size', type=int, default=30000, help='Maximum vocabulary size')
    train_parser.add_argument('--vocab-threshold', type=int, default=5, help='Minimum word frequency')
    train_parser.add_argument('--vocab-max-samples', type=int, default=None, help='Max samples for vocab building')
    train_parser.add_argument('--max-samples', type=int, default=None, help='Maximum training samples')
    train_parser.add_argument('--max-samples-val', type=int, default=None, help='Maximum validation samples')
    train_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    train_parser.add_argument('--keep-n-checkpoints', type=int, default=3, help='Number of checkpoints to keep')
    train_parser.add_argument('--clip-grad-norm', type=float, default=5.0, help='Gradient clipping norm')
    train_parser.add_argument('--plot-every', type=int, default=5, help='Plot training history every N epochs')
    train_parser.add_argument('--rebuild-vocab', action='store_true', help='Rebuild vocabulary')

    # Generation arguments
    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('--image-path', type=str, required=True, help='Path to food image')
    generate_parser.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint')
    generate_parser.add_argument('--output-recipe', type=str, default=None, help='Output recipe file')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()