import torch
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import set_seed
from config import *
from data_utils import load_data, create_dataloaders
from model import load_model_and_tokenizer, evaluate


def visualize_predictions(model, tokenizer, test_loader, device, num_samples=5):
    """Visualize predictions for a few samples."""
    model.eval()

    # Create output directory for plots
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

    samples_visualized = 0
    with torch.no_grad():
        for batch in test_loader:
            if samples_visualized >= num_samples:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Ground truth
            true_intentions = batch["intention"].numpy()
            true_trajectories = batch["future_trajectory"].numpy()

            # Generate complete output
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                pad_token_id=tokenizer.eos_token_id
            )

            # Process each sample in the batch
            for i in range(len(generated)):
                if samples_visualized >= num_samples:
                    break

                # Decode the generated text
                gen_text = tokenizer.decode(generated[i], skip_special_tokens=True)

                # Extract intention
                if "Intention: \"0:" in gen_text:
                    pred_intention = 0
                elif "Intention: \"1:" in gen_text:
                    pred_intention = 1
                elif "Intention: \"2:" in gen_text:
                    pred_intention = 2
                else:
                    pred_intention = -1  # Could not extract

                # Extract trajectory
                pred_trajectory = np.zeros_like(true_trajectories[i])
                try:
                    traj_str = gen_text.split("Trajectory: \"[")[1].split("]\"")[0]
                    points = traj_str.split("), (")
                    points = [p.replace("(", "").replace(")", "") for p in points]

                    for j, point in enumerate(points):
                        if j < len(pred_trajectory):
                            x, y = point.split(",")
                            pred_trajectory[j] = [float(x), float(y)]
                except:
                    pass

                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot trajectories
                ax.plot(true_trajectories[i, :, 0], true_trajectories[i, :, 1], 'b-', label='True Trajectory')
                ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', label='Predicted Trajectory')

                # Add points
                ax.scatter(true_trajectories[i, :, 0], true_trajectories[i, :, 1], c='blue', marker='o')
                ax.scatter(pred_trajectory[:, 0], pred_trajectory[:, 1], c='red', marker='x')

                # Add intention
                intention_map = {0: "Keep lane", 1: "Left lane change", 2: "Right lane change", -1: "Unknown"}
                ax.set_title(f"True: {intention_map[true_intentions[i]]}, Pred: {intention_map[pred_intention]}")

                # Add legend and grid
                ax.legend()
                ax.grid(True)

                # Save the figure
                plt.savefig(os.path.join(OUTPUT_DIR, "visualizations", f"sample_{samples_visualized}.png"))
                plt.close(fig)

                samples_visualized += 1


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    set_seed(42)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and tokenizer (use the best model)
    model_path = os.path.join(OUTPUT_DIR, "best_model")
    if not os.path.exists(model_path):
        logger.info(f"Best model not found at {model_path}, using base model")
        model_path = MODEL_NAME

    logger.info(f"Loading model from: {model_path}")
    model, tokenizer = load_model_and_tokenizer()
    model = model.to(device)

    # Load data
    logger.info(f"Loading data from: {DATA_DIR}")
    _, _, test_data = load_data(DATA_DIR, MAX_SAMPLES)
    logger.info(f"Loaded {len(test_data)} test samples")

    # Create dataloaders
    _, _, test_loader = create_dataloaders([], [], test_data, tokenizer)

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_metrics = evaluate(model, test_loader, tokenizer, device)
    logger.info(f"Test loss: {test_metrics['loss']:.4f}, "
                f"Intention accuracy: {test_metrics['intention_accuracy']:.4f}, "
                f"Trajectory MSE: {test_metrics['trajectory_mse']:.4f}")

    # Visualize some predictions
    logger.info("Visualizing predictions")
    visualize_predictions(model, tokenizer, test_loader, device)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()