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

        # Tokenize the prompt
        tokenized_prompt = self.tokenizer(prompt, padding="max_length", truncation=True,
                                          max_length=MAX_LENGTH, return_tensors="pt")

        # Create input_ids and labels for causal language modeling
        input_ids = tokenized_prompt["input_ids"][0]
        attention_mask = tokenized_prompt["attention_mask"][0]

        # Find the position where the system message ends and the model output begins
        system_msg_end = prompt.find("Thought:")
        if system_msg_end == -1:
            system_msg_end = len(prompt)

        # Create labels: -100 for system message (we don't compute loss on it)
        # and actual token ids for the output tokens
        encoded_system_msg = self.tokenizer(prompt[:system_msg_end], return_tensors="pt")["input_ids"][0]
        system_msg_token_count = len(encoded_system_msg)

        labels = input_ids.clone()
        labels[:system_msg_token_count] = -100  # Don't compute loss on system message

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "intention": intention,
            "future_trajectory": np.array([(frame[2], frame[3]) for frame in future_frames])
            # (delta_y, y_velocity) as trajectory
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

    # Use persistent workers and pin memory for faster data loading
    # but only if not in test mode
    use_persistent_workers = len(train_data) > 10  # Only for larger datasets
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent_workers and BATCH_SIZE > 1,
        num_workers=0 if BATCH_SIZE == 1 else 2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader