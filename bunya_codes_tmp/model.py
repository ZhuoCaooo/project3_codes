import config
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
    """Load the Phi-2 model and tokenizer from Hugging Face with LoRA."""
    print(f"Transformers version: {transformers.__version__}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fix for padding token error
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set padding token to: {tokenizer.pad_token}")
    
    # Determine precision based on config
    dtype = torch.float16 if USE_FP16 else torch.float32
    print(f"Using model precision: {dtype}")

    # Load the base model with memory optimizations
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Apply memory optimizations
    if USE_GRADIENT_CHECKPOINTING:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
    
    if USE_MEMORY_EFFICIENT_ATTENTION:
        print("Enabling memory efficient attention...")
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            print("Disabled model KV-cache")

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

    # Print trainable vs non-trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    return model, tokenizer


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()  # Ensure model is in training mode
    total_loss = 0
    
    # Import custom loss
    import importlib
    import sys
    # Ensure custom_loss is reloaded
    if 'custom_loss' in sys.modules:
        del sys.modules['custom_loss']
    from custom_loss import stable_cross_entropy_loss
    
    progress_bar = tqdm(train_loader, desc="Training")
    for idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Try using the model's built-in loss calculation first
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # If loss is NaN, use custom loss
            if torch.isnan(loss).any():
                print("NaN loss detected, switching to custom loss function")
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
                    scaling=0.5
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
                scaling=0.5
            )

        # Print debugging info for first batch and periodically
        if idx == 0 or idx % 100 == 0:
            print(f"Batch {idx}, Loss: {loss.item()}")
            print(f"Loss requires grad: {loss.requires_grad}")
            
            # Check if model has trainable parameters
            if idx == 0:
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
        
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print("Skipping optimizer step due to NaN gradient")
            continue
        
        if GRADIENT_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                GRADIENT_CLIP_NORM
            )
            
        # Add value clipping in addition to norm clipping
        if hasattr(config, 'CLIP_GRAD_VALUE') and config.CLIP_GRAD_VALUE > 0:
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.data.clamp_(-config.CLIP_GRAD_VALUE, config.CLIP_GRAD_VALUE)

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
    
    # Print memory usage at start
    if torch.cuda.is_available():
        print(f"GPU memory before evaluation: {torch.cuda.memory_allocated()/1e9:.2f}GB")

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
            # Use reduced max_new_tokens to save memory
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS_GENERATION,  # Use the config value
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Use greedy decoding to save memory
            )

            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
