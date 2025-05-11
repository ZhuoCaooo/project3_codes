import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import Counter
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Path to your data directory
data_dir = "output_8sbefore_2safter/"
data_dir = os.path.expanduser(data_dir)  # Expand the ~ to the home directory path

# Feature names based on the dataset description
FEATURE_NAMES = [
    "Left_Lane_Exists", "Right_Lane_Exists", "Delta_Y",
    "Y_Velocity", "Y_Acceleration", "X_Velocity",
    "X_Acceleration", "Vehicle_Type",
    "Preceding_Vehicle_Distance", "Following_Vehicle_Distance",
    "Left_Preceding_Distance", "Left_Alongside_Distance", "Left_Following_Distance",
    "Right_Preceding_Distance", "Right_Alongside_Distance", "Right_Following_Distance"
]

LABEL_NAMES = ["Keep Lane", "Left Lane Change", "Right Lane Change"]


def load_data_sample(file_path):
    """Load a single pickle file and return its contents."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_data_statistics(all_data):
    """Generate statistics about the dataset."""
    total_samples = len(all_data)

    # Count maneuver types
    maneuver_counts = Counter()
    sample_lengths = []

    for data_sample in all_data:
        # Check data structure
        features = data_sample[0] if isinstance(data_sample, tuple) and len(data_sample) > 0 else None
        labels = data_sample[1] if isinstance(data_sample, tuple) and len(data_sample) > 1 else None

        if labels is not None:
            # Get the most common label (the intended maneuver)
            most_common_label = Counter(labels).most_common(1)[0][0]
            maneuver_counts[most_common_label] += 1

        if features is not None:
            sample_lengths.append(len(features))

    return {
        "total_samples": total_samples,
        "maneuver_counts": maneuver_counts,
        "avg_sample_length": np.mean(sample_lengths) if sample_lengths else 0,
        "min_sample_length": min(sample_lengths) if sample_lengths else 0,
        "max_sample_length": max(sample_lengths) if sample_lengths else 0
    }


def visualize_trajectory(data_sample, sample_idx=0, title="Vehicle Trajectory"):
    """Visualize a trajectory from the dataset."""
    # Extract features and labels
    features = data_sample[0] if isinstance(data_sample, tuple) and len(data_sample) > 0 else None
    labels = data_sample[1] if isinstance(data_sample, tuple) and len(data_sample) > 1 else None

    if features is None:
        print("No feature data found")
        return

    # Convert features to a more convenient format
    features_array = np.array(features)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Time steps
    time_steps = np.arange(len(features))

    # Plot the lateral position (Delta Y)
    ax1.plot(time_steps, features_array[:, 2], 'b-', linewidth=2, label='Delta Y (Lateral Position)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Lane center reference line

    # Highlight lane change regions if labels are available
    if labels is not None:
        labels_array = np.array(labels)
        left_changes = labels_array == 1
        right_changes = labels_array == 2

        # Shade regions for left lane changes
        for i in range(1, len(left_changes)):
            if left_changes[i] and not left_changes[i - 1]:
                start_idx = i
            elif not left_changes[i] and left_changes[i - 1]:
                ax1.axvspan(start_idx, i, alpha=0.2, color='g', label='Left Change' if start_idx == i else "")

        # Shade regions for right lane changes
        for i in range(1, len(right_changes)):
            if right_changes[i] and not right_changes[i - 1]:
                start_idx = i
            elif not right_changes[i] and right_changes[i - 1]:
                ax1.axvspan(start_idx, i, alpha=0.2, color='r', label='Right Change' if start_idx == i else "")

    # Add lateral velocity and acceleration
    ax1.plot(time_steps, features_array[:, 3], 'g-', alpha=0.7, label='Y Velocity')
    ax1.plot(time_steps, features_array[:, 4], 'r-', alpha=0.5, label='Y Acceleration')

    ax1.set_title(f"{title} - Sample #{sample_idx}")
    ax1.set_xlabel("Time Steps (25 Hz)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot surrounding vehicle distances in the second subplot
    surrounding_distances = features_array[:, 8:16]  # Features 9-16 are surrounding vehicle distances

    for i, feature_name in enumerate(FEATURE_NAMES[8:16]):
        ax2.plot(time_steps, surrounding_distances[:, i], label=feature_name)

    ax2.set_title("Surrounding Vehicle Distances")
    ax2.set_xlabel("Time Steps (25 Hz)")
    ax2.set_ylabel("Distance")
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_dataset():
    """Main function to analyze the dataset."""
    # List all pickle files
    pickle_files = sorted(glob.glob(os.path.join(data_dir, "*.pickle")))

    if not pickle_files:
        print(f"No pickle files found in {data_dir}")
        return

    print(f"Found {len(pickle_files)} pickle files")

    # Load the first file to explore structure
    first_file = pickle_files[0]
    first_data = load_data_sample(first_file)

    print("\n--- Data Structure Analysis ---")
    print(f"File: {os.path.basename(first_file)}")
    print(f"Data type: {type(first_data)}")
    print(f"Number of samples: {len(first_data)}")

    # Analyze a single sample
    sample = first_data[0]
    print(f"\nSample type: {type(sample)}")

    if isinstance(sample, tuple) and len(sample) >= 2:
        features, labels = sample[0], sample[1]
        print(f"Features type: {type(features)}, Shape: {np.array(features).shape}")
        print(f"Labels type: {type(labels)}, Shape: {np.array(labels).shape}")

        # Show the first few timesteps
        print("\nFirst 5 timesteps of the first sample:")
        for i, (feature, label) in enumerate(zip(features[:5], labels[:5])):
            print(f"Timestep {i}:")
            for j, (name, value) in enumerate(zip(FEATURE_NAMES, feature)):
                print(f"  {name}: {value:.4f}")
            print(f"  Label: {LABEL_NAMES[label]}")

    # Load a subset of files for statistics (e.g., first 5 files)
    num_files_to_analyze = min(5, len(pickle_files))

    print(f"\nAnalyzing first {num_files_to_analyze} files for statistics...")

    all_data = []
    for file_path in tqdm(pickle_files[:num_files_to_analyze]):
        data = load_data_sample(file_path)
        all_data.extend(data)

    # Get statistics
    stats = get_data_statistics(all_data)

    print("\n--- Dataset Statistics ---")
    print(f"Total samples: {stats['total_samples']}")
    print("Maneuver counts:")
    for label_id, count in stats['maneuver_counts'].items():
        print(f"  {LABEL_NAMES[label_id]}: {count} ({count / stats['total_samples'] * 100:.1f}%)")

    print(f"Average sample length: {stats['avg_sample_length']:.1f} timesteps")
    print(f"Min sample length: {stats['min_sample_length']} timesteps")
    print(f"Max sample length: {stats['max_sample_length']} timesteps")

    # Visualize sample trajectories
    print("\n--- Visualizing Sample Trajectories ---")

    # Visualize examples of each maneuver type
    maneuver_examples = {0: None, 1: None, 2: None}

    # Find an example for each maneuver type
    for sample_idx, sample in enumerate(all_data):
        if isinstance(sample, tuple) and len(sample) >= 2:
            labels = sample[1]
            most_common_label = Counter(labels).most_common(1)[0][0]

            # Store the first example of each maneuver type
            if maneuver_examples[most_common_label] is None:
                maneuver_examples[most_common_label] = (sample_idx, sample)

            # Stop once we have an example of each type
            if all(v is not None for v in maneuver_examples.values()):
                break

    # Visualize each maneuver type
    for label_id, example in maneuver_examples.items():
        if example is not None:
            sample_idx, sample = example
            fig = visualize_trajectory(sample, sample_idx, f"{LABEL_NAMES[label_id]} Example")
            plt.savefig(f"trajectory_example_{LABEL_NAMES[label_id].replace(' ', '_')}.png")
            plt.close(fig)

    # Feature distribution analysis
    print("\n--- Feature Distribution Analysis ---")

    # Create a DataFrame for easier analysis
    flat_features = []
    for sample in all_data:
        if isinstance(sample, tuple) and len(sample) >= 2:
            features = sample[0]
            for feature_vec in features:
                flat_features.append(feature_vec)

    feature_df = pd.DataFrame(flat_features, columns=FEATURE_NAMES)

    # Calculate statistics for each feature
    feature_stats = feature_df.describe()
    print(feature_stats)

    # Plot distributions for key features
    key_features = ["Delta_Y", "Y_Velocity", "X_Velocity", "Y_Acceleration", "X_Acceleration"]

    fig, axes = plt.subplots(len(key_features), 1, figsize=(10, 15))

    for i, feature in enumerate(key_features):
        sns.histplot(feature_df[feature], ax=axes[i])
        axes[i].set_title(f"Distribution of {feature}")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_distributions.png")
    plt.close(fig)

    print("\nAnalysis complete! Visualizations saved as PNG files.")


if __name__ == "__main__":
    analyze_dataset()