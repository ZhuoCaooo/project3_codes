import sys
print("Python path at start of model.py:", sys.path)

import transformers
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import os
from config import *
# Add PEFT imports
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer():
    print(f"Transformers version: {transformers.__version__}")
    print("Python path inside load_model_and_tokenizer():", sys.path)
    print(f"Transformers version: {transformers.__version__}")
    
    try:
        # First try to load the config to see if that works
        print("Attempting to load Phi-2 config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"Config loaded successfully. Model type: {config.model_type}")
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        
    """Load the Phi-2 model and tokenizer from Hugging Face with LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fix for padding token error
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to: {tokenizer.pad_token}")

    # Load the base model without gradient checkpointing
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use fp16 precision to save memory
        trust_remote_code=True  # NEW: Add this parameter
    )

    # Apply LoRA
    if USE_LORA:
        print("Setting up LoRA configuration...")
        # Configure LoRA
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            task_type=TaskType.CAUSAL_LM
        )

        print("Applying LoRA to model...")
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)

        # Make sure we're explicitly enabling training mode
        model.train()

        # Print trainable parameters
        model.print_trainable_parameters()

    return model, tokenizer


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()  # Ensure model is in training mode
    total_loss = 0

    # Import custom loss function
    import importlib
    import sys
    # Force reload custom_loss module
    if 'custom_loss' in sys.modules:
        del sys.modules['custom_loss']
    from custom_loss import stable_cross_entropy_loss

    progress_bar = tqdm(train_loader, desc="Training")
    for idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Print diagnostics for the first few batches
        if idx < 3:
            valid_count = (labels != -100).sum().item()
            total_count = labels.numel()
            print(f"Batch {idx}: {valid_count}/{total_count} valid labels ({valid_count / total_count:.2%})")

        # Clear gradients
        optimizer.zero_grad()

        # Try standard loss first
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # If loss is 0 or NaN, use custom loss
            if loss.item() == 0 or torch.isnan(loss).any():
                print(f"Zero or NaN loss detected, switching to custom loss function")
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = stable_cross_entropy_loss(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.1,
                    scaling=1.0
                )
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print("Using custom loss function")
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = stable_cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
                scaling=1.0
            )

        # Print debugging info for first batch
        if idx == 0:
            print(f"Loss on first batch: {loss.item()}")
            print(f"Loss requires grad: {loss.requires_grad}")
            # Check if model has trainable parameters
            has_trainable = False
            for name, param in model.named_parameters():
                if param.requires_grad:
                    has_trainable = True
                    print(f"Trainable parameter: {name}, shape: {param.shape}")
                    break
            if not has_trainable:
                print("WARNING: No trainable parameters found!")

        # Handle gradient accumulation
        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        loss.backward()

        # Update weights if needed
        if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or idx == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Update progress bar
        total_loss += loss.item() * (1 if GRADIENT_ACCUMULATION_STEPS <= 1 else GRADIENT_ACCUMULATION_STEPS)
        progress_bar.set_postfix({"loss": total_loss / (idx + 1)})

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

    # For LoRA models, we need to save the adapter separately
    if hasattr(model, "save_pretrained") and USE_LORA:
        # Save the adapter
        model.save_pretrained(output_dir)
        # Save the tokenizer
        tokenizer.save_pretrained(output_dir)
        print(f"LoRA adapter saved to {output_dir}")
    else:
        # Standard save for full models
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
