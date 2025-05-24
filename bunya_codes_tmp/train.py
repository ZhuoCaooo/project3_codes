#!/usr/bin/env python3
"""
Fixed LC-LLM Training Script based on literature implementation
Addresses both memory and loss=0.0 issues
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_int8_training
)
import numpy as np
import re
import gc


class FixedLCLLMTrainer:
    def __init__(self, model_name="microsoft/phi-2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """Load model with proper quantization like literature"""
        
        # Set CUDA memory settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            padding_side='left'  # Like literature
        )
        
        # Fix padding token like literature
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model with 8-bit quantization...")
        
        # Use 8-bit quantization like literature
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map={"": 0},  # Single GPU like literature
            load_in_8bit=True,  # Key: 8-bit quantization
            low_cpu_mem_usage=True
        )

        # Prepare for 8-bit training like literature
        self.model = prepare_model_for_int8_training(self.model)

        # LoRA config matching literature values
        lora_config = LoraConfig(
            r=64,                    # Literature value
            lora_alpha=16,           # Literature value
            target_modules=[         # Phi-2 equivalent of literature targets
                "q_proj", "k_proj", "v_proj", "dense",
                "fc1", "fc2"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        print("✅ Model loaded with 8-bit quantization and LoRA")

    def fix_data_format(self, text):
        """Convert to proper instruction format like literature"""
        # Remove problematic tokens
        text = text.replace("<s>", "").replace("</s>", "").strip()
        
        # Convert to simple instruction format
        if "[INST]" in text and "[/INST]" in text:
            parts = text.split("[/INST]")
            if len(parts) >= 2:
                input_part = parts[0].replace("[INST]", "").replace("<<SYS>>", "").replace("<</SYS>>", "").strip()
                response_part = parts[1].strip()
                
                # Simple format that works better
                return f"<|im_start|>system\nYou are an expert driving prediction model.<|im_end|>\n<|im_start|>user\n{input_part}<|im_end|>\n<|im_start|>assistant\n{response_part}<|im_end|>"
        
        return text

    def load_and_process_data(self, train_file, val_file=None):
        """Load and process data with proper formatting"""
        print(f"Loading data from: {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        # Process and fix data
        processed_data = []
        for i, sample in enumerate(train_data):
            try:
                original_text = sample["text"]
                fixed_text = self.fix_data_format(original_text)
                
                if len(fixed_text.strip()) > 100:
                    processed_data.append({"text": fixed_text})
                    
            except Exception as e:
                print(f"Skipping sample {i}: {e}")
                continue

        print(f"Processed {len(processed_data)} samples")
        
        # Split data
        if val_file and os.path.exists(val_file):
            with open(val_file, 'r') as f:
                val_data_raw = json.load(f)
            val_data = [{"text": self.fix_data_format(s["text"])} for s in val_data_raw[:500]]  # Limit val size
        else:
            split_idx = int(len(processed_data) * 0.9)
            val_data = processed_data[split_idx:]
            processed_data = processed_data[:split_idx]

        print(f"Final split - Train: {len(processed_data)}, Val: {len(val_data)}")
        
        return Dataset.from_list(processed_data), Dataset.from_list(val_data)

    def tokenize_function(self, examples):
        """Proper tokenization with instruction tuning like literature"""
        print(f"Tokenizing {len(examples['text'])} samples...")
        
        # Process each text to separate input/output for training
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for text in examples["text"]:
            # Find the assistant response part
            if "<|im_start|>assistant\n" in text:
                parts = text.split("<|im_start|>assistant\n")
                if len(parts) == 2:
                    input_text = parts[0] + "<|im_start|>assistant\n"
                    response_text = parts[1].replace("<|im_end|>", "")
                    full_text = input_text + response_text
                else:
                    full_text = text
                    input_text = text[:len(text)//2]
                    response_text = text[len(text)//2:]
            else:
                # Fallback
                full_text = text
                input_text = text[:len(text)//2]
                response_text = text[len(text)//2:]

            # Tokenize full text
            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=512,  # Reduced for memory
                padding=False,
                return_tensors=None
            )
            
            # Tokenize input part to find where to mask
            input_tokens = self.tokenizer(
                input_text,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None
            )
            
            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]
            
            # Create labels - mask input part, keep response part
            labels = input_ids.copy()
            input_length = min(len(input_tokens["input_ids"]), len(labels))
            
            # Mask input part (don't train on it)
            for i in range(input_length):
                labels[i] = -100
                
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)

        print(f"✅ Tokenized {len(model_inputs['input_ids'])} samples")
        return model_inputs

    def train(self, train_dataset, val_dataset, output_dir="./outputs"):
        """Training with proper arguments like literature"""
        
        print("Processing datasets...")
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=100,  # Small batches for memory
            remove_columns=["text"],
            desc="Tokenizing training data"
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=["text"],
            desc="Tokenizing validation data"
        )

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

        os.makedirs(output_dir, exist_ok=True)

        # Training arguments matching literature approach
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Literature training schedule
            num_train_epochs=2,
            learning_rate=5e-4,           # Literature value
            warmup_steps=300,             # Reduced proportionally
            
            # Memory optimized batch sizes
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Literature uses 8
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            
            # Memory optimizations
            fp16=True,                    # Literature uses bf16, we use fp16
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            
            # Logging
            logging_steps=10,
            logging_dir=f"{output_dir}/logs",
            
            # Disable problematic features
            remove_unused_columns=False,
            report_to=None,
            push_to_hub=False,
        )

        # Data collator like literature
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        print("🚀 Starting training with fixed configuration...")
        
        try:
            # Clear cache before training
            torch.cuda.empty_cache()
            
            # Train
            trainer.train()
            
            # Save final model
            final_model_path = os.path.join(output_dir, "final_model")
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            print(f"✅ Training completed! Model saved to {final_model_path}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            torch.cuda.empty_cache()
            raise e

        return trainer


def main():
    """Main training function"""
    try:
        print("🚀 Starting Fixed LC-LLM Training")
        print("=" * 60)
        
        # Memory settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        trainer = FixedLCLLMTrainer(model_name="microsoft/phi-2")
        trainer.load_model_and_tokenizer()
        
        # Load data
        train_dataset, val_dataset = trainer.load_and_process_data(
            "../data/phi2_training_data.json",
            "../data/phi2_testing_data.json"
        )
        
        print(f"✅ Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

        # Train
        trained_model = trainer.train(
            train_dataset, 
            val_dataset, 
            output_dir="./outputs_fixed"
        )
        
        print("🎉 Training pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()