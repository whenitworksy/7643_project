import os
import torch
import json
import time
from datetime import datetime


class CheckpointManager:
    """
    Handles saving and loading of model checkpoints
    """

    def __init__(self, checkpoint_dir, model, optimizer, scheduler=None, keep_n_latest=3):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.keep_n_latest = keep_n_latest
        self.metadata_path = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize or load checkpoint metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load checkpoint metadata if it exists"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'last_epoch': 0,
                'best_val_loss': float('inf'),
                'checkpoints': []
            }

    def _save_metadata(self):
        """Save checkpoint metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(self, epoch, val_loss=None, is_best=False, extra_data=None):
        """
        Save model checkpoint

        Args:
            epoch: Current training epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            extra_data: Additional data to save in checkpoint
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }

        # Add scheduler if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Add extra data if provided
        if extra_data is not None:
            checkpoint.update(extra_data)

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checkpoint_info = {
            'path': checkpoint_name,
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': timestamp,
            'is_best': is_best
        }

        # Add to list of checkpoints
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['last_epoch'] = epoch

        # Update best val loss if this is the best model
        if is_best and val_loss is not None:
            self.metadata['best_val_loss'] = val_loss

            # Save a copy as best model
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

        # Keep only N latest checkpoints
        if len(self.metadata['checkpoints']) > self.keep_n_latest:
            # Sort by epoch
            sorted_checkpoints = sorted(
                self.metadata['checkpoints'],
                key=lambda x: x['epoch'],
                reverse=True
            )

            # Keep best model and N latest
            keep_paths = set([c['path'] for c in sorted_checkpoints[:self.keep_n_latest]])
            keep_paths.add('best_model.pth')  # Always keep best model

            # Remove old checkpoints
            for checkpoint_info in sorted_checkpoints[self.keep_n_latest:]:
                path = checkpoint_info['path']
                if path not in keep_paths:
                    full_path = os.path.join(self.checkpoint_dir, path)
                    if os.path.exists(full_path):
                        os.remove(full_path)

            # Update metadata to reflect removals
            self.metadata['checkpoints'] = sorted_checkpoints[:self.keep_n_latest]

        # Save updated metadata
        self._save_metadata()

        return checkpoint_path

    def load_latest_checkpoint(self):
        """
        Load the latest checkpoint

        Returns:
            epoch: Last training epoch
            checkpoint: Full checkpoint data
        """
        if not self.metadata['checkpoints']:
            print("No checkpoints found.")
            return 0, None

        # Sort by epoch and get latest
        sorted_checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['epoch'],
            reverse=True
        )

        latest = sorted_checkpoints[0]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest['path'])

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} not found.")
            return 0, None

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint['epoch'], checkpoint

    def load_best_checkpoint(self):
        """
        Load the best checkpoint based on validation loss

        Returns:
            checkpoint: Full checkpoint data
        """
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')

        if not os.path.exists(best_path):
            print("No best model checkpoint found.")
            return None

        # Load checkpoint
        checkpoint = torch.load(best_path)

        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore scheduler if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded best model checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint