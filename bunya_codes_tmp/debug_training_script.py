#!/usr/bin/env python3
"""
DEBUGGING VERSION - Find why loss is still 0.0
This will show us exactly what's happening with data and tokenization
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

def debug_data_processing(data_file, num_samples_to_check=5):
    """Debug the data loading and processing"""
    logger.info("🔍 DEBUGGING: Loading and examining raw data...")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Total samples in file: {len(data)}")
    
    # Check first few samples
    for i in range(min(num_samples_to_check, len(data))):
        sample = data[i]
        text = sample["text"]
        
        logger.info(f"\n--- SAMPLE {i+1} ---")
        logger.info(f"Full text length: {len(text)} characters")
        
        # Check if ### Assistant: exists
        if "### Assistant:" in text:
            parts = text.split("### Assistant:")
            input_part = parts[0] + "### Assistant:"
            target_part = parts[1].strip() if len(parts) > 1 else ""
            
            logger.info(f"✅ Found ### Assistant: delimiter")
            logger.info(f"Input part length: {len(input_part)} characters")
            logger.info(f"Target part length: {len(target_part)} characters")
            logger.info(f"Target preview: {target_part[:100]}...")
            
            if len(target_part) == 0:
                logger.warning("❌ TARGET PART IS EMPTY!")
        else:
            logger.warning(f"❌ No ### Assistant: found in sample {i+1}")
    
    return data

def debug_tokenization(tokenizer, input_text, target_text, sample_id):
    """Debug a single tokenization example"""
    logger.info(f"\n🔍 DEBUGGING TOKENIZATION - Sample {sample_id}")
    
    # Tokenize parts separately
    input_tokens = tokenizer(input_text, truncation=False, padding=False, add_special_tokens=False)
    target_tokens = tokenizer(target_text, truncation=False, padding=False, add_special_tokens=False)
    
    logger.info(f"Input tokens: {len(input_tokens['input_ids'])}")
    logger.info(f"Target tokens: {len(target_tokens['input_ids'])}")
    
    # Show some actual token IDs
    logger.info(f"First 10 input token IDs: {input_tokens['input_ids'][:10]}")
    logger.info(f"First 10 target token IDs: {target_tokens['input_ids'][:10]}")
    
    # Combine and create labels
    full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]
    input_length = len(input_tokens["input_ids"])
    
    labels = full_input_ids.copy()
    for i in range(input_length):
        labels[i] = -100
    
    # Count non-masked labels
    non_masked_labels = [l for l in labels if l != -100]
    logger.info(f"Total tokens: {len(labels)}")
    logger.info(f"Masked tokens (-100): {len(labels) - len(non_masked_labels)}")
    logger.info(f"Non-masked tokens (target): {len(non_masked_labels)}")
    
    if len(non_masked_labels) == 0:
        logger.warning("❌ ALL TOKENS ARE MASKED! No target tokens to learn from!")
        return False
    
    logger.info(f"✅ {len(non_masked_labels)} target tokens to learn from")
    return True

def load_and_process_data(data_file, max_samples=None):
    """Load and process data with debugging"""
    logger.info("Loading data...")
    
    # First debug the raw data
    data = debug_data_processing(data_file)
    
    if max_samples:
        data = data[:max_samples]
    
    processed = []
    failed_samples = 0
    
    for i, item in enumerate(data):
        text = item["text"]
        
        # Split on ### Assistant:
        if "### Assistant:" in text:
            parts = text.split("### Assistant:")
            if len(parts) >= 2:
                input_text = parts[0] + "### Assistant:"
                target_text = parts[1].strip()
                
                # Debug first few samples
                if i < 3:
                    has_targets = debug_tokenization(
                        AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True),
                        input_text, target_text, i+1
                    )
                    if not has_targets:
                        logger.warning(f"Sample {i+1} has no target tokens!")
                
                if len(target_text) > 0:
                    processed.append({
                        "input": input_text,
                        "target": target_text
                    })
                else:
                    failed_samples += 1
                    logger.warning(f"Skipping sample {i+1}: empty target")
        else:
            failed_samples += 1
            logger.warning(f"Skipping sample {i+1}: no ### Assistant: delimiter")
    
    logger.info(f"Successfully processed: {len(processed)} samples")
    logger.info(f"Failed samples: {failed_samples}")
    
    if len(processed) == 0:
        raise RuntimeError("❌ NO VALID SAMPLES PROCESSED!")
    
    return Dataset.from_list(processed)

def precise_tokenization_with_debug(examples, tokenizer, max_length=1200):
    """Tokenization with extensive debugging"""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    total_samples = len(examples["input"])
    samples_with_targets = 0
    total_target_tokens = 0
    
    for i, (input_text, target_text) in enumerate(zip(examples["input"], examples["target"])):
        # Tokenize parts
        input_tokens = tokenizer(input_text, truncation=False, padding=False, add_special_tokens=False)
        target_tokens = tokenizer(target_text, truncation=False, padding=False, add_special_tokens=False)
        
        # Combine
        full_input_ids = input_tokens["input_ids"] + target_tokens["input_ids"]
        full_attention_mask = input_tokens["attention_mask"] + target_tokens["attention_mask"]
        
        # Handle truncation
        if len(full_input_ids) > max_length:
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
        
        # Create labels
        labels = full_input_ids.copy()
        for j in range(min(input_length, len(labels))):
            labels[j] = -100
        
        # Count target tokens for this sample
        sample_target_tokens = sum(1 for l in labels if l != -100)
        if sample_target_tokens > 0:
            samples_with_targets += 1
            total_target_tokens += sample_target_tokens
        
        model_inputs["input_ids"].append(full_input_ids)
        model_inputs["attention_mask"].append(full_attention_mask)
        model_inputs["labels"].append(labels)
    
    # Debug summary
    logger.info(f"🔍 TOKENIZATION SUMMARY:")
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Samples with target tokens: {samples_with_targets}")
    logger.info(f"   Total target tokens: {total_target_tokens}")
    logger.info(f"   Avg target tokens per sample: {total_target_tokens/max(1,samples_with_targets):.1f}")
    
    if samples_with_targets == 0:
        raise RuntimeError("❌ NO SAMPLES HAVE TARGET TOKENS!")
    
    return model_inputs

def main():
    """Main function with extensive debugging"""
    logger.info("🚀 Starting DEBUGGING VERSION")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("✅ Tokenizer loaded")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    logger.info("✅ Base model loaded")
    
    # Create and apply LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    
    # Verify parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_trainable = any(p.requires_grad for p in model.parameters())
    logger.info(f"🔍 Parameter verification:")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Has trainable params: {has_trainable}")
    
    if not has_trainable:
        raise RuntimeError("❌ NO TRAINABLE PARAMETERS!")
    
    # Load and debug data - SMALL SAMPLE FIRST
    logger.info("🔍 Testing with small sample first...")
    dataset = load_and_process_data("../data/phi2_training_data.json", max_samples=50)
    
    # Split train/val
    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Apply tokenization with debugging
    logger.info("🔍 Applying tokenization with debugging...")
    train_dataset = train_dataset.map(
        lambda x: precise_tokenization_with_debug(x, tokenizer),
        batched=True,
        batch_size=20,
        remove_columns=["input", "target"]
    )
    
    val_dataset = val_dataset.map(
        lambda x: precise_tokenization_with_debug(x, tokenizer),
        batched=True,
        batch_size=20,
        remove_columns=["input", "target"]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs_phi2_debug",
        num_train_epochs=1,  # Just 1 epoch for debugging
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Smaller for debugging
        evaluation_strategy="steps",
        eval_steps=5,  # More frequent evaluation
        save_steps=10,
        logging_steps=1,  # Log every step
        logging_first_step=True,
        fp16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=None,
        optim="adamw_torch",
        weight_decay=0.01,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Debug optimizer
    optimizer = trainer.create_optimizer()
    param_groups = optimizer.param_groups
    total_optimizer_params = sum(len(group['params']) for group in param_groups)
    
    logger.info(f"🔍 Final verification:")
    logger.info(f"   Optimizer parameter groups: {len(param_groups)}")
    logger.info(f"   Total parameters in optimizer: {total_optimizer_params}")
    if len(param_groups) > 0:
        logger.info(f"   Learning rate: {param_groups[0]['lr']}")
        logger.info(f"   Weight decay: {param_groups[0].get('weight_decay', 'N/A')}")
    
    if total_optimizer_params == 0:
        raise RuntimeError("❌ OPTIMIZER HAS NO PARAMETERS!")
    
    # Check a single batch manually
    logger.info("🔍 Checking single batch manually...")
    train_dataloader = trainer.get_train_dataloader()
    batch = next(iter(train_dataloader))
    
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Labels shape: {batch['labels'].shape}")
    
    # Count non-masked labels in batch
    labels = batch['labels']
    non_masked = (labels != -100).sum().item()
    total_tokens = labels.numel()
    
    logger.info(f"Tokens in batch: {total_tokens}")
    logger.info(f"Non-masked tokens: {non_masked}")
    logger.info(f"Masked tokens: {total_tokens - non_masked}")
    
    if non_masked == 0:
        raise RuntimeError("❌ ALL TOKENS IN BATCH ARE MASKED!")
    
    logger.info(f"✅ {non_masked} target tokens available for training")
    
    # Start training
    logger.info("🎯 Starting debug training...")
    trainer.train()
    
    logger.info("✅ Debug training completed!")

if __name__ == "__main__":
    main()