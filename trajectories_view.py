import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_all_trajectories():
    """Load all trajectory data from pickle files"""
    trajectories = []
    directions = []

    for i in range(1, 61):
        filename = f"output_4sbefore_4safter/result{i:02d}.pickle"
        if Path(filename).exists():
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                for trajectory, direction in data:
                    trajectories.append(trajectory)
                    directions.append(direction)

    return trajectories, directions


def extract_feature_sequences(trajectories, directions, feature_idx):
    """Extract specific feature sequences by lane change type"""
    left_changes = []
    right_changes = []
    lane_keeping = []

    for traj, dir_seq in zip(trajectories, directions):
        feature_seq = [frame[feature_idx] for frame in traj]

        if dir_seq[0] == 1:  # Left lane change
            left_changes.append(feature_seq)
        elif dir_seq[0] == 2:  # Right lane change
            right_changes.append(feature_seq)
        elif dir_seq[0] == 0:  # Lane keeping
            lane_keeping.append(feature_seq)

    return left_changes, right_changes, lane_keeping


def plot_trajectories():
    """Visualize lane change trajectories"""
    print("Loading trajectory data...")
    trajectories, directions = load_all_trajectories()

    # Feature indices (from your construct_features function)
    DELTA_Y = 2  # Difference from lane center
    Y_VELOCITY = 3  # Lateral velocity
    Y_POSITION = 6  # Absolute Y position

    # Create time axis (200 frames at 25 Hz = 8 seconds)
    time_axis = np.linspace(-4, 4, 200)  # -4s to +4s, boundary at t=0

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Lane Change Trajectory Analysis', fontsize=16)

    # Extract feature sequences
    left_delta, right_delta, keep_delta = extract_feature_sequences(trajectories, directions, DELTA_Y)
    left_vel, right_vel, keep_vel = extract_feature_sequences(trajectories, directions, Y_VELOCITY)

    # Plot 1: Delta Y (distance from lane center)
    ax1 = axes[0, 0]
    for seq in right_delta[:50]:  # Show first 50 for clarity
        ax1.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_delta[:50]:
        ax1.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)

    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Boundary Crossing')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance from Lane Center (m)')
    ax1.set_title('Lateral Position Relative to Lane Center')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Right LC', 'Left LC', 'Boundary'])

    # Plot 2: Y Velocity (lateral velocity)
    ax2 = axes[0, 1]
    for seq in right_vel[:50]:
        ax2.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_vel[:50]:
        ax2.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)

    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Lateral Velocity (m/s)')
    ax2.set_title('Lateral Velocity')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Right LC Analysis - Delta Y distribution at different time points
    ax3 = axes[1, 0]
    if right_delta:
        right_array = np.array(right_delta)
        # Show distribution at boundary crossing (frame 100)
        boundary_values = right_array[:, 100]
        pre_values = right_array[:, 50]  # 2s before boundary
        post_values = right_array[:, 150]  # 2s after boundary

        ax3.hist(pre_values, bins=30, alpha=0.5, label='2s before', color='orange')
        ax3.hist(boundary_values, bins=30, alpha=0.5, label='At boundary', color='red')
        ax3.hist(post_values, bins=30, alpha=0.5, label='2s after', color='darkred')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Distance from Lane Center (m)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Right LC: Distance Distribution Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Right LC Analysis - Velocity direction consistency
    ax4 = axes[1, 1]
    if right_vel:
        right_vel_array = np.array(right_vel)

        # Calculate mean and std of velocity profiles
        mean_vel = np.mean(right_vel_array, axis=0)
        std_vel = np.std(right_vel_array, axis=0)

        ax4.fill_between(time_axis, mean_vel - std_vel, mean_vel + std_vel,
                         alpha=0.3, color='red', label='±1 std')
        ax4.plot(time_axis, mean_vel, 'r-', linewidth=2, label='Mean velocity')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Boundary')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Lateral Velocity (m/s)')
        ax4.set_title('Right LC: Mean Velocity Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Left lane changes: {len(left_delta)}")
    print(f"Right lane changes: {len(right_delta)}")
    print(f"Lane keeping: {len(extract_feature_sequences(trajectories, directions, DELTA_Y)[2])}")

    if right_vel:
        right_vel_array = np.array(right_vel)
        # Check velocity direction consistency at boundary
        boundary_velocities = right_vel_array[:, 100]  # At boundary crossing
        positive_vel = np.sum(boundary_velocities > 0)
        negative_vel = np.sum(boundary_velocities < 0)
        zero_vel = np.sum(np.abs(boundary_velocities) < 0.1)

        print(f"\nRight Lane Change Velocity Direction at Boundary:")
        print(f"Positive velocity (moving away from lane center): {positive_vel}")
        print(f"Negative velocity (moving toward lane center): {negative_vel}")
        print(f"Near-zero velocity: {zero_vel}")
        print(f"Direction consistency: {max(positive_vel, negative_vel) / len(boundary_velocities) * 100:.1f}%")

    plt.show()


if __name__ == "__main__":
    plot_trajectories()