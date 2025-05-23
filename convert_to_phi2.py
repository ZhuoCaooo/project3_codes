#!/usr/bin/env python3
"""
Convert Llama-2 format data to Phi-2 compatible format
"""

import json
import re


def convert_llama_to_phi2_format(input_file, output_file):
    """Convert Llama-2 chat format to Phi-2 format"""

    with open(input_file, 'r') as f:
        data = json.load(f)

    converted_data = []

    for i, sample in enumerate(data):
        original_text = sample["text"]

        # Extract the actual content from Llama-2 format
        # Original: <s>[INST] <<SYS>>\n[SYSTEM]\n<</SYS>>\n\n[USER] [/INST] [RESPONSE] </s>

        # Step 1: Remove outer tags
        text = original_text.replace("<s>", "").replace("</s>", "").strip()

        # Step 2: Extract system message
        sys_match = re.search(r'\[INST\] <<SYS>>\n(.*?)\n<</SYS>>\n\n(.*?) \[/INST\] (.*)', text, re.DOTALL)

        if sys_match:
            system_msg = sys_match.group(1).strip()
            user_msg = sys_match.group(2).strip()
            response = sys_match.group(3).strip()

            # Convert to Phi-2 format (simpler instruction format)
            phi2_text = f"""### System:
{system_msg}

### Human:
{user_msg}

### Assistant:
{response}"""

        else:
            # Fallback: simple extraction
            print(f"Warning: Could not parse sample {i}, using fallback")

            # Try to extract basic content
            if "[/INST]" in text:
                parts = text.split("[/INST]")
                if len(parts) >= 2:
                    input_part = parts[0].replace("[INST]", "").replace("<<SYS>>", "").replace("<</SYS>>", "").strip()
                    response_part = parts[1].strip()

                    phi2_text = f"""### Human:
{input_part}

### Assistant:
{response_part}"""
                else:
                    continue  # Skip malformed samples
            else:
                continue  # Skip malformed samples

        # Create new sample
        converted_sample = {
            "text": phi2_text
        }
        converted_data.append(converted_sample)

        # Debug: print first few conversions
        if i < 3:
            print(f"\n--- Sample {i} Conversion ---")
            print("ORIGINAL (first 200 chars):")
            print(original_text[:200] + "...")
            print("\nCONVERTED:")
            print(phi2_text[:300] + "...")

    # Save converted data
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Original samples: {len(data)}")
    print(f"Converted samples: {len(converted_data)}")
    print(f"Saved to: {output_file}")


def test_phi2_tokenization(file_path, tokenizer):
    """Test if converted data tokenizes properly"""

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"\nTesting tokenization on {len(data)} samples...")

    max_token_id = 0
    problematic_samples = 0

    for i, sample in enumerate(data[:10]):  # Test first 10 samples
        try:
            tokens = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=1024)
            input_ids = tokens["input_ids"][0]

            sample_max = input_ids.max().item()
            max_token_id = max(max_token_id, sample_max)

            if sample_max >= tokenizer.vocab_size:
                problematic_samples += 1
                print(f"Sample {i}: Max token ID {sample_max} >= vocab size {tokenizer.vocab_size}")

        except Exception as e:
            print(f"Sample {i}: Tokenization error: {e}")
            problematic_samples += 1

    print(f"Overall max token ID: {max_token_id}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Problematic samples: {problematic_samples}/10")

    return problematic_samples == 0


if __name__ == "__main__":
    # Convert both your files
    print("Converting training data...")
    convert_llama_to_phi2_format("lcllm_training_data.json", "phi2_training_data.json")

    print("\nConverting testing data...")
    convert_llama_to_phi2_format("lcllm_testing_data.json", "phi2_testing_data.json")

    # Test with Phi-2 tokenizer
    print("\nTesting converted data...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    success = test_phi2_tokenization("phi2_training_data.json", tokenizer)

    if success:
        print("✅ Conversion successful! Use phi2_training_data.json for training.")
    else:
        print("❌ Still have tokenization issues. Need further fixes.")