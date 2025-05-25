#!/usr/bin/env python3
"""
Check actual token lengths in your converted data
"""

import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


def analyze_token_lengths(file_path):
    """Analyze actual token lengths in your data"""

    print("🔍 ANALYZING TOKEN LENGTHS")
    print("=" * 50)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load your data
    with open(file_path, 'r') as f:
        data = json.load(f)

    token_lengths = []
    sample_texts = []

    print("Analyzing token lengths...")

    for i, sample in enumerate(data[:100]):  # Check first 100 samples
        text = sample["text"]

        # Tokenize
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        length = tokens["input_ids"].shape[1]
        token_lengths.append(length)

        if i < 5:  # Store first 5 for detailed analysis
            sample_texts.append((text, length))

    # Statistics
    token_lengths = np.array(token_lengths)

    print(f"\n📊 TOKEN LENGTH STATISTICS")
    print("-" * 30)
    print(f"Mean length: {token_lengths.mean():.1f}")
    print(f"Median length: {np.median(token_lengths):.1f}")
    print(f"Min length: {token_lengths.min()}")
    print(f"Max length: {token_lengths.max()}")
    print(f"95th percentile: {np.percentile(token_lengths, 95):.1f}")
    print(f"99th percentile: {np.percentile(token_lengths, 99):.1f}")

    # Check different max_length options
    print(f"\n📋 TRUNCATION ANALYSIS")
    print("-" * 30)

    for max_len in [256, 512, 1024, 1536, 2048]:
        truncated = np.sum(token_lengths > max_len)
        percent = (truncated / len(token_lengths)) * 100
        print(f"Max length {max_len}: {truncated}/{len(token_lengths)} samples truncated ({percent:.1f}%)")

    # Show sample breakdowns
    print(f"\n🔬 SAMPLE BREAKDOWN")
    print("-" * 30)

    for i, (text, length) in enumerate(sample_texts[:3]):
        print(f"\nSample {i + 1} ({length} tokens):")

        # Break down by section
        if "### System:" in text:
            parts = text.split("### Human:")
            if len(parts) >= 2:
                system_part = parts[0]
                human_part = "### Human:" + parts[1].split("### Assistant:")[0] if "### Assistant:" in parts[1] else \
                parts[1]
                assistant_part = "### Assistant:" + parts[1].split("### Assistant:")[1] if "### Assistant:" in parts[
                    1] else ""

                sys_tokens = len(tokenizer(system_part)["input_ids"])
                human_tokens = len(tokenizer(human_part)["input_ids"])
                assistant_tokens = len(tokenizer(assistant_part)["input_ids"]) if assistant_part else 0

                print(f"  System: {sys_tokens} tokens")
                print(f"  Human: {human_tokens} tokens")
                print(f"  Assistant: {assistant_tokens} tokens")

        # Show first 200 chars
        print(f"  Preview: {text[:200]}...")

    # Memory usage estimation
    print(f"\n💾 MEMORY IMPACT")
    print("-" * 30)

    batch_size = 1
    for max_len in [256, 512, 1024]:
        # Rough memory estimation (very approximate)
        # Each token ~2 bytes for IDs, plus attention matrix
        memory_mb = (max_len * max_len * batch_size * 2) / (1024 * 1024)  # Attention matrix
        memory_mb += (max_len * batch_size * 2) / (1024 * 1024)  # Token IDs

        print(f"Max length {max_len}, batch size {batch_size}: ~{memory_mb:.1f} MB per sample")

    # Recommendation
    print(f"\n🎯 RECOMMENDATION")
    print("-" * 30)

    # Find optimal length that keeps 95% of data
    optimal_length = int(np.percentile(token_lengths, 95))

    if optimal_length <= 512:
        print(f"✅ GOOD NEWS: 95% of your samples fit in {optimal_length} tokens")
        print(f"✅ Recommended MAX_LENGTH: 512 (safe choice)")
        print(f"✅ Memory usage will be reasonable")
    elif optimal_length <= 1024:
        print(f"⚠️  95% of your samples need {optimal_length} tokens")
        print(f"⚠️  Recommended MAX_LENGTH: 1024 (may need more GPU memory)")
        print(f"⚠️  Consider truncation strategy")
    else:
        print(f"❌ 95% of your samples need {optimal_length} tokens")
        print(f"❌ This is quite long - consider:")
        print(f"   - Shortening system messages")
        print(f"   - Reducing surrounding vehicle details")
        print(f"   - Using MAX_LENGTH: 1024 with truncation")

    return token_lengths, optimal_length


def plot_token_distribution(token_lengths):
    """Plot token length distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=256, color='red', linestyle='--', label='Current MAX_LENGTH (256)')
    plt.axvline(x=512, color='orange', linestyle='--', label='Alternative (512)')
    plt.axvline(x=1024, color='green', linestyle='--', label='Alternative (1024)')
    plt.axvline(x=np.percentile(token_lengths, 95), color='blue', linestyle='-', label='95th percentile')

    plt.xlabel('Token Length')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Token Lengths in Your Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('token_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Plot saved as 'token_length_distribution.png'")


if __name__ == "__main__":
    # Analyze your Phi-2 converted data
    try:
        token_lengths, optimal = analyze_token_lengths("phi2_training_data.json")
        plot_token_distribution(token_lengths)

        print(f"\n🏁 FINAL VERDICT")
        print("=" * 50)
        print(f"Your current MAX_LENGTH: 256")
        print(f"Recommended MAX_LENGTH: {min(1024, optimal)}")
        print(f"Memory impact: {'Low' if optimal <= 512 else 'Medium' if optimal <= 1024 else 'High'}")

    except FileNotFoundError:
        print("❌ File 'phi2_training_data.json' not found!")
        print("   Run your convert_to_phi2.py script first.")
    except Exception as e:
        print(f"❌ Error: {e}")