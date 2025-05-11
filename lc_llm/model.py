import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import os
from config import *


def load_model_and_tokenizer():
    """Load the Phi-2 model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fix for padding token error
    if tokenizer.pad_token is None:
        # Set pad_token to eos_token for Phi-2
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to: {tokenizer.pad_token}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    return model, tokenizer


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass with gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (progress_bar.n + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or progress_bar.n == len(train_loader) - 1:
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update progress bar
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

    return total_loss / len(train_loader)


def evaluate(model, eval_loader, tokenizer, device):
    """Evaluate the model on validation or test data."""
    model.eval()
    total_loss = 0

    # Metrics for intention prediction
    correct_intentions = 0
    total_samples = 0

    # Metrics for trajectory prediction
    trajectory_mse = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Ground truth
            true_intentions = batch["intention"].to(device)
            true_trajectories = batch["future_trajectory"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate complete output for intention and trajectory extraction
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id
            )

            # Extract intention and trajectory from generated text
            for i in range(len(generated)):
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)

                # Extract intention (this is simplified and would need more robust extraction)
                if "Intention: \"0:" in gen_text:
                    pred_intention = 0
                elif "Intention: \"1:" in gen_text:
                    pred_intention = 1
                elif "Intention: \"2:" in gen_text:
                    pred_intention = 2
                else:
                    pred_intention = -1  # Could not extract

                # Count correct intentions
                if pred_intention == true_intentions[i].item():
                    correct_intentions += 1
                total_samples += 1

                # Extract trajectory (simplified)
                # In a real implementation, you would use regex or more robust extraction
                if "Trajectory: \"[" in gen_text:
                    try:
                        # Extract the trajectory string and convert to points
                        traj_str = gen_text.split("Trajectory: \"[")[1].split("]\"")[0]
                        points = traj_str.split("), (")
                        points = [p.replace("(", "").replace(")", "") for p in points]

                        pred_trajectory = []
                        for point in points:
                            x, y = point.split(",")
                            pred_trajectory.append([float(x), float(y)])

                        pred_trajectory = np.array(pred_trajectory)

                        # Calculate MSE with true trajectory
                        if len(pred_trajectory) == len(true_trajectories[i]):
                            mse = np.mean((pred_trajectory - true_trajectories[i].cpu().numpy()) ** 2)
                            trajectory_mse += mse
                    except:
                        # Could not extract trajectory
                        pass

    intention_accuracy = correct_intentions / total_samples if total_samples > 0 else 0
    avg_trajectory_mse = trajectory_mse / total_samples if total_samples > 0 else float('inf')

    return {
        "loss": total_loss / len(eval_loader),
        "intention_accuracy": intention_accuracy,
        "trajectory_mse": avg_trajectory_mse
    }


def save_model(model, tokenizer, output_dir):
    """Save the model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")