import os
import sys
print("Python path at start of test_train.py:", sys.path)
print("Current working directory:", os.getcwd())

import logging
import torch
from transformers import set_seed

# Use test config instead of regular config
os.environ["USE_TEST_CONFIG"] = "1"
from test_config import *

# Import other modules
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
    
    logger.info("STARTING TEST JOB")
    logger.info(f"Using test configuration: MAX_SAMPLES={MAX_SAMPLES}, NUM_EPOCHS={NUM_EPOCHS}")

    # Set seed for reproducibility
    set_seed(42)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load model and tokenizer
        logger.info(f"Loading model: {MODEL_NAME}")
        model, tokenizer = load_model_and_tokenizer()
        model = model.to(device)
        logger.info("Model loaded successfully")

        # Load data
        logger.info(f"Loading data from: {DATA_DIR}")
        train_data, val_data, test_data = load_data(DATA_DIR, MAX_SAMPLES)
        logger.info(f"Loaded {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")

        # Create dataloaders (just use a small subset for testing)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data[:MAX_SAMPLES//2], 
            val_data[:MAX_SAMPLES//4], 
            test_data[:MAX_SAMPLES//4], 
            tokenizer
        )
        logger.info("DataLoaders created successfully")

        # Setup optimizer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": WEIGHT_DECAY,
            }
        ]

        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
        )
        logger.info("Optimizer and scheduler initialized")

        # Do a mini training run (only 5 batches)
        logger.info("Starting mini training run (5 batches)...")
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Print diagnostic information
            logger.info(f"CONFIG CHECK: MAX_LENGTH={MAX_LENGTH}, BATCH_SIZE={BATCH_SIZE}")
            logger.info(f"INPUT SHAPE: {input_ids.shape}, attention_mask.shape={attention_mask.shape}")
            if torch.cuda.is_available():
                logger.info(f"MEMORY: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            logger.info(f"Batch {batch_idx+1}/5 completed with loss: {loss.item()}")
        
        # Test evaluation
        logger.info("Testing evaluation...")
        eval_metrics = evaluate(model, val_loader, tokenizer, device)
        logger.info(f"Evaluation results: {eval_metrics}")
        
        # Test model saving
        logger.info("Testing model saving...")
        save_model(model, tokenizer, os.path.join(OUTPUT_DIR, "test_model"))
        logger.info("Model saved successfully")
        
        logger.info("TEST JOB COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
