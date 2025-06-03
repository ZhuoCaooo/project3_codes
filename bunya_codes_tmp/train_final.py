#!/usr/bin/env python3
"""
Final working LC-LLM training script - FP16 only, literature tokenization
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model


def load_and_process_data(data_file):
    """Load and process data with literature approach"""
    print("Loading data...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        text = item["text"]
        
        # Split on ### Assistant: (your format)
        if "### Assistant:" in text:
            parts = text.split("### Assistant:")
            if len(parts) >= 2:
                input_text = parts[0] + "### Assistant:"
                target_text = parts[1].strip()
                
                processed.append({
                    "input": input_text,
                    "target": target_text
                })
    
    print(f"Processed {len(processed)} samples")
    return Dataset.from_list(processed)


def tokenize_function(examples, tokenizer, max_length=512):
    """Literature-style tokenization"""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for input_text, target_text in zip(examples["input"], examples["target"]):
        # Combine input + target
        full_text = input_text + target_text
        
        # Tokenize full text
        full_tokens = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        
        # Tokenize input part to find where to mask
        input_tokens = tokenizer(input_text, truncation=True, max_length=max_length, padding=False)
        
        # Create labels with masking
        labels = full_tokens["input_ids"].copy()
        input_length = len(input_tokens["input_ids"])
        
        # Mask input part (literature approach)
        for i in range(min(input_length, len(labels))):
            labels[i] = -100
        
        model_inputs["input_ids"].append(full_tokens["input_ids"])
        model_inputs["attention_mask"].append(full_tokens["attention_mask"])
        model_inputs["labels"].append(labels)
    
    return model_inputs


def main():
    print("🚀 Starting Final LC-LLM Training")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (FP16 only, no quantization)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # LoRA config
    lora_config = LoraConfig(
        r=64,  # Literature value
        lora_alpha=16,  # Literature value
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and process data
    dataset = load_and_process_data("../data/phi2_training_data.json")
    
    # Split train/val
    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    
    # Tokenize
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=50,
        remove_columns=["input", "target"]
    )
    
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=50,
        remove_columns=["input", "target"]
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs_final_4epoches",
        num_train_epochs=4,
        learning_rate=5e-4,  # Literature value
        warmup_steps=100,    # Reduced for stability
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Literature value
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=10,
        fp16=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model("./outputs_final_4epochs/final_model")
    print("✅ Training complete!")


if __name__ == "__main__":
    main()