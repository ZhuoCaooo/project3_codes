#!/usr/bin/env python3
"""
Clean Phi-2 LC-LLM training script - No quantization dependencies
Includes all key improvements: precise tokenization, optimized LoRA
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
    This is the KEY improvement from the paper
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
        
        # Step 4: Create labels with precise masking (CRITICAL IMPROVEMENT)
        labels = full_input_ids.copy()
        
        # Mask input part with -100 (literature approach)
        for i in range(min(input_length, len(labels))):
            labels[i] = -100
        
        model_inputs["input_ids"].append(full_input_ids)
        model_inputs["attention_mask"].append(full_attention_mask)
        model_inputs["labels"].append(labels)
    
    return model_inputs

def main():
    """Main training function"""
    logger.info("🚀 Starting Clean Phi-2 LC-LLM Training")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("✅ Tokenizer loaded")
    
    # Load model (clean FP16)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    logger.info("✅ Model loaded with FP16")
    
    # Create optimized LoRA config
    lora_config = LoraConfig(
        r=32,  # Literature: reduced for stability
        lora_alpha=64,  # Literature: 2x rank ratio
        target_modules=[
            # Attention modules
            "q_proj", "k_proj", "v_proj", "dense",
            # Phi-2 MLP modules (key improvement)
            "fc1", "fc2"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("✅ LoRA adapters added")
    
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
    
    # Training arguments (literature optimized)
    training_args = TrainingArguments(
        output_dir="./outputs_phi2_clean",
        num_train_epochs=3,
        learning_rate=2e-4,  # Literature value
        warmup_steps=200,
        
        # Batch strategy for long sequences
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 16
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        
        # Logging
        logging_steps=25,
        logging_first_step=True,
        
        # Performance
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Bunya compatible
        dataloader_num_workers=0,     # Bunya compatible
        
        # Other
        remove_unused_columns=False,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save final model
    final_save_path = "./phi2_lc_llm_clean_final"
    trainer.save_model(final_save_path)
    logger.info(f"✅ Training completed! Model saved to: {final_save_path}")
    
    logger.info("📊 Training Summary:")
    logger.info(f"- Clean FP16 training (no quantization issues)")
    logger.info(f"- Precise tokenization with exact masking")
    logger.info(f"- Optimized LoRA: r=32, alpha=64")
    logger.info(f"- Phi-2 specific modules: fc1, fc2")
    logger.info(f"- Literature learning rate: 2e-4")

if __name__ == "__main__":
    main()