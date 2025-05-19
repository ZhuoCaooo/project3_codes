import torch
import torch.nn.functional as F


def stable_cross_entropy_loss(logits, labels, ignore_index=-100, label_smoothing=0.0, scaling=1.0):
    """Stable cross entropy loss function that handles edge cases better."""
    # Check if we have any valid labels
    valid_indices = (labels != ignore_index).sum().item()

    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Logits has NaN: {torch.isnan(logits).any()}")
    print(f"Labels has NaN: {torch.isnan(labels).any()}")
    print(f"Valid labels count: {valid_indices}")

    if valid_indices == 0:
        print("WARNING: No valid labels found!")
        # Return a small non-zero loss with grad to prevent issues
        dummy_loss = torch.tensor(0.01, device=logits.device, requires_grad=True)
        return dummy_loss

    # Regular cross entropy with smoothing
    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction='mean'
    )

    # Scale the loss
    scaled_loss = loss * scaling
    print(f"Raw loss: {loss.item()}, Scaled loss: {scaled_loss.item()}")

    return scaled_loss