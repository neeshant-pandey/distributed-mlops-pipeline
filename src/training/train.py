"""
Main training script for DeepLOB model.
"""

import argparse
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pathlib import Path

from src.data.dataset import LOBDataModule
from src.models.deeplob import create_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.metrics import compute_metrics, get_classification_report

import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepLOB model")

    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml", help="Training config path"
    )
    parser.add_argument(
        "--model-config", type=str, default="config/model_config.yaml", help="Model config path"
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (override config)")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (override config)"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (override config)")
    parser.add_argument(
        "--horizon", type=int, default=0, help="Prediction horizon index (0-4 for k=1,2,3,5,10)"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--experiment-name", type=str, default="lob-midprice-prediction", help="MLflow experiment name"
    )
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configs
    print("Loading configuration...")
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)

    # Override with command line args
    if args.epochs:
        train_config["training"]["epochs"] = args.epochs
    if args.batch_size:
        train_config["training"]["batch_size"] = args.batch_size
    if args.lr:
        train_config["training"]["learning_rate"] = args.lr
    if args.device:
        train_config["training"]["device"] = args.device

    # Set device
    device = train_config["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = "cpu"

    print(f"Using device: {device}")

    # Set random seeds
    seed = train_config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize MLflow
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run(run_name=args.run_name)

        # Log parameters
        mlflow.log_params({
            "horizon": args.horizon,
            "horizon_k": [1, 2, 3, 5, 10][args.horizon],
            "epochs": train_config["training"]["epochs"],
            "batch_size": train_config["training"]["batch_size"],
            "learning_rate": train_config["training"]["learning_rate"],
            "weight_decay": train_config["training"]["weight_decay"],
            "window_size": model_config["model"]["input_shape"]["width"],
            "device": device,
            "seed": seed,
        })

    print(f"\n{'='*70}")
    print("Training Configuration:")
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Horizon: k={[1, 2, 3, 5, 10][args.horizon]}")
    print(f"  Epochs: {train_config['training']['epochs']}")
    print(f"  Batch size: {train_config['training']['batch_size']}")
    print(f"  Learning rate: {train_config['training']['learning_rate']}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    # Create data module
    print("Loading data...")
    data_module = LOBDataModule(
        data_path=args.data_dir,
        horizon_index=args.horizon,
        window_size=model_config["model"]["input_shape"]["width"],
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["training"]["num_workers"],
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type="deeplob",
        n_features=model_config["model"]["input_shape"]["height"],
        n_classes=model_config["model"]["num_classes"],
        conv_filters=model_config["model"]["conv_layers"][0]["filters"],
        inception_filters=model_config["model"]["inception"]["filters"],
        lstm_hidden_size=model_config["model"]["lstm"]["hidden_size"],
        lstm_num_layers=model_config["model"]["lstm"]["num_layers"],
    )

    print(f"Model parameters: {model.get_num_params():,}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["training"]["learning_rate"],
        weight_decay=train_config["training"]["weight_decay"],
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_config["training"]["scheduler"]["step_size"],
        gamma=train_config["training"]["scheduler"]["gamma"],
    )

    # Create loss criterion (with class weights if needed)
    # Get class weights from dataset
    class_weights = data_module.train_dataset.dataset.get_class_weights()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f"Class weights: {class_weights.numpy()}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=train_config["training"]["checkpoint_dir"],
        use_mlflow=use_mlflow,
    )

    # Train
    history = trainer.train(
        n_epochs=train_config["training"]["epochs"],
        early_stopping_patience=train_config["training"]["early_stopping"]["patience"],
        save_every=train_config["training"]["save_every_n_epochs"],
        scheduler=scheduler,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = data_module.test_dataloader()

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Compute test metrics
    test_metrics = compute_metrics(
        np.array(all_targets), np.array(all_preds)
    )

    print("\nTest Set Results:")
    print("="*70)
    for key, value in test_metrics.items():
        print(f"{key:25}: {value:.4f}")
    print("="*70)

    print("\nTest Set Classification Report:")
    print(get_classification_report(
        np.array(all_targets), np.array(all_preds)
    ))

    # Log test metrics to MLflow
    if use_mlflow:
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # End run
        mlflow.end_run()

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
