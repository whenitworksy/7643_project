import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch


class TrainingLogger:
    """
    Handles logging during training
    """

    def __init__(self, log_dir, model_name="food2recipe"):
        self.log_dir = log_dir
        self.model_name = model_name
        self.start_time = time.time()

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
        self.history_file = os.path.join(log_dir, f"{model_name}_{timestamp}_history.json")

        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'learning_rate': [],
            'time_per_epoch': [],
        }

        # Write initial log entry
        with open(self.log_file, 'w') as f:
            f.write(f"# Training log for {model_name}\n")
            f.write(f"# Started at {timestamp}\n\n")

    def log_epoch(self, epoch, train_loss, val_loss=None, lr=None, samples_per_sec=None,
                  examples=None, extra_metrics=None):
        """
        Log metrics for an epoch
        """
        elapsed = time.time() - self.start_time
        epoch_time = elapsed / (epoch + 1)  # Average time per epoch

        # Update history
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if lr is not None:
            self.history['learning_rate'].append(lr)
        self.history['time_per_epoch'].append(epoch_time)

        # Add extra metrics if provided
        if extra_metrics:
            for key, value in extra_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

        # Format log message
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        log_message = f"Epoch {epoch:4d} | Time: {time_str} | Train Loss: {train_loss:.4f}"

        if val_loss is not None:
            log_message += f" | Val Loss: {val_loss:.4f}"

        if lr is not None:
            log_message += f" | LR: {lr:.6f}"

        if samples_per_sec is not None:
            log_message += f" | Samples/sec: {samples_per_sec:.2f}"

        if extra_metrics:
            for key, value in extra_metrics.items():
                log_message += f" | {key}: {value:.4f}"

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")

            # Log examples if provided
            if examples:
                f.write("\nExamples:\n")
                for i, (img_path, target, prediction) in enumerate(examples):
                    f.write(f"Example {i + 1}:\n")
                    f.write(f"  Image: {img_path}\n")
                    f.write(f"  Target: {target}\n")
                    f.write(f"  Prediction: {prediction}\n\n")

        # Save history to JSON
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Print to console
        print(log_message)

    def plot_history(self, save_path=None):
        """
        Plot training history
        """
        epochs = self.history['epochs']

        plt.figure(figsize=(12, 5))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot learning rate
        if 'learning_rate' in self.history and self.history['learning_rate']:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.history['learning_rate'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path)

        return plt.gcf()