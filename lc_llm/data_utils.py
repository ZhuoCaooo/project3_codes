import pickle
import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
import prompt_utils


class LaneChangeDataset(Dataset):
    def __init__(self, data_samples, tokenizer):
        self.data_samples = data_samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # Extract features and label
        features, labels = sample[0], sample[1]

        # Get the current frame (we use the middle frame as the current frame)
        current_frame_idx = len(features) // 2
        current_frame = features[current_frame_idx]

        # Get historical trajectory (10 frames before current frame)
        history_frames = features[max(0, current_frame_idx - 10):current_frame_idx]

        # Get future trajectory (4 seconds after current frame)
        # Assuming 25 Hz sampling, 4 seconds = 100 frames
        future_frame_indices = [current_frame_idx + 25, current_frame_idx + 50,
                                current_frame_idx + 75, current_frame_idx + 100]
        future_frames = [features[i] if i < len(features) else features[-1] for i in future_frame_indices]

        # Get lane change intention (most common label in the next 4 seconds)
        future_labels = labels[current_frame_idx:current_frame_idx + 100]
        if len(future_labels) > 0:
            from collections import Counter
            intention = Counter(future_labels).most_common(1)[0][0]
        else:
            intention = 0  # Default to keep lane

        # Create prompt for this sample
        prompt = prompt_utils.create_prompt(current_frame, history_frames, intention, future_frames)

        # After tokenizing the prompt:
        tokenized_prompt = self.tokenizer(prompt, padding="max_length", truncation=True,
                                          max_length=MAX_LENGTH, return_tensors="pt")

        input_ids = tokenized_prompt["input_ids"][0]
        attention_mask = tokenized_prompt["attention_mask"][0]

        # REPLACE the existing label creation code with this improved version:
        # Find output section more reliably
        output_marker = "Thought:"
        output_start = prompt.find(output_marker)

        # Fallback options if Thought: is not found
        if output_start == -1:
            output_marker = "Final Answer:"
            output_start = prompt.find(output_marker)

        # Final fallback - use a percentage split
        if output_start == -1:
            # Split at 70% of the prompt
            output_start = int(len(prompt) * 0.7)
            print(f"Warning: Couldn't find output markers in sample {idx}, using 70/30 split")

        # Tokenize just the part before the output section
        input_text = prompt[:output_start]
        input_tokens = self.tokenizer(input_text, return_tensors="pt")["input_ids"][0]
        input_token_count = len(input_tokens)

        # Safety check to avoid index errors
        input_token_count = min(input_token_count, len(input_ids))

        # Create labels: -100 for input portion, actual ids for output portion
        labels = input_ids.clone()
        labels[:input_token_count] = -100

        # Diagnostic check - ensure we have some valid labels
        valid_label_count = (labels != -100).sum().item()
        if valid_label_count == 0:
            print(f"Warning: Sample {idx} has ZERO valid labels! Using 30% fallback.")
            # Emergency fix: Use the last 30% of tokens for training
            fallback_split = int(len(labels) * 0.7)
            labels[fallback_split:] = input_ids[fallback_split:].clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "intention": intention,
            "future_trajectory": np.array([(frame[2], frame[3]) for frame in future_frames])
        }


def load_data(data_dir, max_samples=None):
    """Load and preprocess the lane change dataset."""
    pickle_files = sorted(glob.glob(os.path.join(data_dir, "*.pickle")))

    all_data = []
    for file_path in pickle_files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            all_data.extend(data)

        if max_samples and len(all_data) >= max_samples:
            all_data = all_data[:max_samples]
            break

    # Split into train, validation, and test
    total_samples = len(all_data)
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)

    # Shuffle data for random split
    np.random.shuffle(all_data)

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data, tokenizer):
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    train_dataset = LaneChangeDataset(train_data, tokenizer)
    val_dataset = LaneChangeDataset(val_data, tokenizer)
    test_dataset = LaneChangeDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader