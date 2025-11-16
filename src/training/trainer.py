"""
Trainer class for DeepLOB model with MLflow integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np

from src.utils.metrics import MetricsTracker, get_classification_report


class Trainer:
    """Trainer for DeepLOB model with MLflow tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        use_mlflow: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_mlflow: Whether to use MLflow tracking
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.current_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        metrics_tracker = MetricsTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            metrics_tracker.update(outputs, targets, loss.item())

            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute epoch metrics
        metrics = metrics_tracker.compute()
        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        metrics_tracker = MetricsTracker()

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]  "
            )

            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Update metrics
                metrics_tracker.update(outputs, targets, loss.item())

        # Compute metrics
        metrics = metrics_tracker.compute()

        # Print classification report
        print("\nValidation Classification Report:")
        print(get_classification_report(
            np.array(metrics_tracker.targets),
            np.array(metrics_tracker.predictions)
        ))

        return metrics

    def train(
        self,
        n_epochs: int,
        early_stopping_patience: int = None,
        save_every: int = 5,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.

        Args:
            n_epochs: Number of epochs
            early_stopping_patience: Early stopping patience (None = disabled)
            save_every: Save checkpoint every N epochs
            scheduler: Learning rate scheduler

        Returns:
            Dictionary of training history
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        patience_counter = 0

        print(f"\n{'='*70}")
        print(f"Starting Training: {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*70}\n")

        for epoch in range(n_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["learning_rates"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_acc": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_acc": val_metrics["accuracy"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{n_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | "
                  f"Val Acc:   {val_metrics['accuracy']:.4f}")

            # Learning rate scheduler
            if scheduler:
                scheduler.step()

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt")
                patience_counter = 0
                print(f"  ✓ Best model saved (acc: {self.best_val_acc:.4f})")
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered (patience: {early_stopping_patience})")
                break

            print("-" * 70)

        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best Val Acc: {self.best_val_acc:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

        return history

    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_artifact(str(save_path))

    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        load_path = self.checkpoint_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
