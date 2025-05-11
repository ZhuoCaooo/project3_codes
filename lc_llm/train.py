import torch
import os
import logging
from transformers import set_seed
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from config import *
from data_utils import load_data, create_dataloaders
from model import load_model_and_tokenizer, train_epoch, evaluate, save_model


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    set_seed(42)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and tokenizer with LoRA if enabled
    logger.info(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer()

    if USE_LORA:
        logger.info(f"Using LoRA with rank={LORA_RANK}, alpha={LORA_ALPHA}")

    model = model.to(device)

    # Load data
    logger.info(f"Loading data from: {DATA_DIR}")
    train_data, val_data, test_data = load_data(DATA_DIR, MAX_SAMPLES)
    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, tokenizer)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, tokenizer, device)
        logger.info(f"Validation loss: {val_metrics['loss']:.4f}, "
                    f"Intention accuracy: {val_metrics['intention_accuracy']:.4f}, "
                    f"Trajectory MSE: {val_metrics['trajectory_mse']:.4f}")

        # Save model if it's the best so far
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
            save_model(model, tokenizer, os.path.join(OUTPUT_DIR, "best_model"))

        # Save checkpoint for this epoch
        save_model(model, tokenizer, os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}"))

    # Final evaluation on test set
    logger.info("Evaluating on test set")
    test_metrics = evaluate(model, test_loader, tokenizer, device)
    logger.info(f"Test loss: {test_metrics['loss']:.4f}, "
                f"Intention accuracy: {test_metrics['intention_accuracy']:.4f}, "
                f"Trajectory MSE: {test_metrics['trajectory_mse']:.4f}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()