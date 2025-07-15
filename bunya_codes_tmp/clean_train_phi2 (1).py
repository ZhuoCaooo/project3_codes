#!/usr/bin/env python3
"""
FIXED Phi-2 LC-LLM training script
Key fix: Let PEFT handle parameter management automatically
"""

import torch
import json
import logging
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_data(data_file, max_samples=None):
    """Load and process data with literature approach"""
    logger.info("Loading data...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    processed = []
    for item in data:
        text = item["text"]
        
        # Split on ### Assistant: (your clean format)
        if "### Assistant:" in text:
            parts = text.split("### Assistant:")
            if len(parts) >= 2:
                input_text = parts[0] + "### Assistant:"
                target_text = parts[1].strip()
                
                processed.append({
                    "input": input_text,
                    "target": target_text
                })
        else:
            logger.warning(f"Skipping sample without ### Assistant: delimiter")
    
    logger.info(f"Successfully processed {len(processed)} samples")
    return Dataset.from_list(processed)

def precise_tokenization(examples, tokenizer, max_length=1200):
    """
    Literature-style precise tokenization with exact masking
    """
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for input_text, target_text in zip(examples["input"], examples["target"]):
        # Step 1: Tokenize input part separately for exact token count
        input_tokens = tokenizer(
            input_text, 
            truncation=False, 
            padding=False,
            add_special_tokens=False
        )
        
        # Step 2: Tokenize target part separately  
        target_tokens = tokenizer(
            target_text,
            truncation=False,
            padding=False, 
            add_special_tokens=False
        )
        
        # Step 3: Combine input + target
        full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]
        full_attention_mask = input_tokens["attention_mask"] + target_tokens["attention_mask"]
        
        # Truncate if too long
        if len(full_input_ids) > max_length:
            # Prioritize keeping target, truncate input if needed
            if len(target_tokens["input_ids"]) < max_length - 50:
                input_keep = max_length - len(target_tokens["input_ids"])
                full_input_ids = input_tokens["input_ids"][-input_keep:] + target_tokens["input_ids"]
                full_attention_mask = input_tokens["attention_mask"][-input_keep:] + target_tokens["attention_mask"]
                input_length = input_keep
            else:
                full_input_ids = full_input_ids[:max_length]
                full_attention_mask = full_attention_mask[:max_length]
                input_length = len(input_tokens["input_ids"])
        else:
            input_length = len(input_tokens["input_ids"])
        
        # Step 4: Create labels with precise masking
        labels = full_input_ids.copy()
        
        # Mask input part with -100
        for i in range(min(input_length, len(labels))):
            labels[i] = -100
        
        model_inputs["input_ids"].append(full_input_ids)
        model_inputs["attention_mask"].append(full_attention_mask)
        model_inputs["labels"].append(labels)
    
    return model_inputs

def main():
    """Main training function with FIXED parameter management"""
    logger.info("🚀 Starting FIXED Phi-2 LC-LLM Training")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("✅ Tokenizer loaded")
    
    # Load model in FP16 - NO MANUAL PARAMETER FREEZING!
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    logger.info("✅ Base model loaded")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "dense",  # Attention layers
            "fc1", "fc2"  # MLP layers for Phi-2
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    # Apply LoRA - PEFT handles parameter freezing automatically!
    model = get_peft_model(model, lora_config)
    
    # ✅ CRITICAL FIX: Don't manually freeze/unfreeze anything!
    # PEFT has already:
    # 1. Frozen all base model parameters
    # 2. Made LoRA parameters trainable
    # 3. Set up proper gradient flow
    
    # Verify the setup
    model.print_trainable_parameters()
    
    # Ensure training mode
    model.train()
    
    # ✅ VERIFICATION: Check that parameters are properly set up
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🔍 Verification:")
    logger.info(f"   Trainable: {trainable_params:,}")
    logger.info(f"   Total: {total_params:,}")
    logger.info(f"   Percentage: {100 * trainable_params / total_params:.2f}%")
    
    # ✅ CRITICAL: Verify some parameters actually require gradients
    has_trainable = any(p.requires_grad for p in model.parameters())
    if not has_trainable:
        raise RuntimeError("❌ NO TRAINABLE PARAMETERS FOUND!")
    logger.info("✅ LoRA setup verified - parameters ready for training")
    
    # Load and process data
    dataset = load_and_process_data("../data/phi2_training_data.json")
    
    # Split train/val
    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Apply precise tokenization
    logger.info("Applying precise tokenization...")
    train_dataset = train_dataset.map(
        lambda x: precise_tokenization(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=["input", "target"]
    )
    
    val_dataset = val_dataset.map(
        lambda x: precise_tokenization(x, tokenizer),
        batched=True, 
        batch_size=100,
        remove_columns=["input", "target"]
    )
    
    logger.info("✅ Precise tokenization completed")
    
    # ✅ FIXED: Training arguments with proper learning rate schedule
    training_args = TrainingArguments(
        output_dir="./outputs_phi2_fixed",
        num_train_epochs=3,
        
        # ✅ CRITICAL FIX: Proper learning rate setup
        learning_rate=2e-4,
        lr_scheduler_type="cosine",  # Better than linear decay
        warmup_ratio=0.03,  # Use ratio instead of fixed steps
        
        # Batch strategy
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        
        # Logging - more frequent to catch issues
        logging_steps=10,  # More frequent logging
        logging_first_step=True,
        
        # Performance
        fp16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        
        # Other
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # ✅ CRITICAL: Ensure proper optimizer setup
        optim="adamw_torch",  # Explicit optimizer
        weight_decay=0.01,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8
    )
    
    # ✅ Create trainer - PEFT model is ready!
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # ✅ VERIFICATION: Check trainer sees the parameters
    optimizer = trainer.create_optimizer()
    param_groups = optimizer.param_groups
    total_optimizer_params = sum(len(group['params']) for group in param_groups)
    logger.info(f"🔍 Optimizer verification:")
    logger.info(f"   Parameter groups: {len(param_groups)}")
    logger.info(f"   Total parameters in optimizer: {total_optimizer_params}")
    logger.info(f"   Learning rate: {param_groups[0]['lr']}")
    
    if total_optimizer_params == 0:
        raise RuntimeError("❌ OPTIMIZER HAS NO PARAMETERS!")
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save final model
    final_save_path = "./phi2_lc_llm_fixed_final"
    trainer.save_model(final_save_path)
    logger.info(f"✅ Training completed! Model saved to: {final_save_path}")

if __name__ == "__main__":
    main()