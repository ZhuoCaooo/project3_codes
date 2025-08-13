import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D  # For custom legends


def load_all_trajectories():
    """Load all trajectory data from pickle files"""
    trajectories = []
    directions = []

    # Assuming pickle files are in a folder named 'output_4sbefore_4safter'
    folder = Path("output_4sbefore_4safter")
    if not folder.exists():
        print(f"Error: Directory not found -> {folder.resolve()}")
        print("Please ensure the pickle files from your 'run' script are in this directory.")
        return [], []

    for i in range(1, 61):
        filename = folder / f"result{i:02d}.pickle"
        if filename.exists():
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
        # Ensure the trajectory has the expected length
        if len(traj) != 200:
            continue

        feature_seq = [frame[feature_idx] for frame in traj]

        if dir_seq[0] == 1:  # Left lane change
            left_changes.append(feature_seq)
        elif dir_seq[0] == 2:  # Right lane change
            right_changes.append(feature_seq)
        elif dir_seq[0] == 0:  # Lane keeping
            lane_keeping.append(feature_seq)

    return left_changes, right_changes, lane_keeping


def plot_trajectories():
    """Visualize lane change trajectories for both lateral and longitudinal motion"""
    print("Loading trajectory data...")
    trajectories, directions = load_all_trajectories()

    if not trajectories:
        print("No trajectory data loaded. Exiting.")
        return

    # Feature indices from your construct_features function
    DELTA_Y = 2  # Difference from lane center
    Y_VELOCITY = 3  # Lateral velocity
    Y_ACCELERATION = 4  # Lateral acceleration
    X_POSITION = 5  # Absolute X position
    X_VELOCITY = 7  # Longitudinal velocity

    # Create time axis (200 frames at 25 Hz = 8 seconds)
    time_axis = np.linspace(-4, 4, 200)  # -4s to +4s, boundary at t=0

    # ### CHANGE: 2x4 grid instead of 2x3 ###
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # <-- CHANGE THIS LINE
    fig.suptitle('Lane Change Trajectory Analysis (Lateral & Longitudinal)', fontsize=18, y=0.99)

    # --- Extract feature sequences ---
    left_delta, right_delta, _ = extract_feature_sequences(trajectories, directions, DELTA_Y)
    left_vel, right_vel, _ = extract_feature_sequences(trajectories, directions, Y_VELOCITY)
    left_accel, right_accel, _ = extract_feature_sequences(trajectories, directions, Y_ACCELERATION)
    left_x_pos, right_x_pos, _ = extract_feature_sequences(trajectories, directions, X_POSITION)
    left_x_vel, right_x_vel, _ = extract_feature_sequences(trajectories, directions, X_VELOCITY)

    # --- Plot 1: Delta Y (distance from lane center) ---
    ax1 = axes[0, 0]
    for seq in right_delta[:100]:
        ax1.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_delta[:100]:
        ax1.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)

    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance from Lane Center (m)')
    ax1.set_title('Lateral Position Relative to Lane Center')
    ax1.grid(True, alpha=0.3)
    legend_elements_ax1 = [Line2D([0], [0], color='r', lw=2, label='Right LC'),
                           Line2D([0], [0], color='b', lw=2, label='Left LC'),
                           Line2D([0], [0], color='black', linestyle='--', label='Boundary')]
    ax1.legend(handles=legend_elements_ax1)

    # --- Plot 2: Y Velocity (lateral velocity) ---
    ax2 = axes[0, 1]
    for seq in right_vel[:100]:
        ax2.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_vel[:100]:
        ax2.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)

    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Lateral Velocity (m/s)')
    ax2.set_title('Lateral Velocity')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Y Acceleration (lateral acceleration) ---
    ax3 = axes[0, 2]
    for seq in right_accel[:100]:
        ax3.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_accel[:100]:
        ax3.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)

    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Lateral Acceleration (m/s²)')
    ax3.set_title('Lateral Acceleration')
    ax3.grid(True, alpha=0.3)

    # --- NEW Plot 4: X Velocity (longitudinal velocity) ---  # <-- ADD THIS ENTIRE BLOCK
    ax4 = axes[0, 3]
    for seq in right_x_vel[:100]:
        ax4.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_x_vel[:100]:
        ax4.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)

    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Longitudinal Velocity (m/s)')
    ax4.set_title('Longitudinal Velocity')
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: Right LC Analysis - Delta Y distribution ---
    ax5 = axes[1, 0]
    if right_delta:
        right_array = np.array(right_delta)
        ax5.hist(right_array[:, 50], bins=30, alpha=0.6, label='t = -2s', color='orange', density=True)
        ax5.hist(right_array[:, 100], bins=30, alpha=0.6, label='t = 0s (Boundary)', color='red', density=True)
        ax5.hist(right_array[:, 150], bins=30, alpha=0.6, label='t = +2s', color='darkred', density=True)
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Distance from Lane Center (m)')
        ax5.set_ylabel('Density')
        ax5.set_title('Right LC: Distribution of Lateral Position')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # --- Plot 6: Right LC Analysis - Mean Lateral Velocity Profile ---
    ax6 = axes[1, 1]
    if right_vel:
        right_vel_array = np.array(right_vel)
        mean_vel = np.mean(right_vel_array, axis=0)
        std_vel = np.std(right_vel_array, axis=0)

        ax6.fill_between(time_axis, mean_vel - std_vel, mean_vel + std_vel, alpha=0.3, color='red', label='±1 std')
        ax6.plot(time_axis, mean_vel, 'r-', linewidth=2, label='Mean Velocity')
        ax6.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Boundary')
        ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Lateral Velocity (m/s)')
        ax6.set_title('Right LC: Mean Lateral Velocity Profile')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # --- Plot 7: Longitudinal Displacement ---
    ax7 = axes[1, 2]
    for seq in right_x_pos[:100]:
        normalized_seq = np.array(seq) - seq[0]  # Normalize to start at 0
        ax7.plot(time_axis, normalized_seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_x_pos[:100]:
        normalized_seq = np.array(seq) - seq[0]
        ax7.plot(time_axis, normalized_seq, 'b-', alpha=0.3, linewidth=0.5)

    ax7.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Relative Longitudinal Position (m)')
    ax7.set_title('Longitudinal Displacement (Normalized at t=-4s)')
    ax7.grid(True, alpha=0.3)
    legend_elements_ax7 = [Line2D([0], [0], color='r', lw=2, label='Right LC'),
                           Line2D([0], [0], color='b', lw=2, label='Left LC')]
    ax7.legend(handles=legend_elements_ax7)

    # --- NEW Plot 8: Mean Longitudinal Velocity Profile ---  # <-- ADD THIS ENTIRE BLOCK
    ax8 = axes[1, 3]
    if right_x_vel:
        right_x_vel_array = np.array(right_x_vel)
        mean_x_vel = np.mean(right_x_vel_array, axis=0)
        std_x_vel = np.std(right_x_vel_array, axis=0)

        ax8.fill_between(time_axis, mean_x_vel - std_x_vel, mean_x_vel + std_x_vel,
                         alpha=0.3, color='red', label='±1 std')
        ax8.plot(time_axis, mean_x_vel, 'r-', linewidth=2, label='Mean Velocity')
        ax8.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Boundary')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Longitudinal Velocity (m/s)')
        ax8.set_title('Right LC: Mean Longitudinal Velocity Profile')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # --- Print statistics (keep existing and add X velocity stats) ---
    print(f"\nDataset Statistics:")
    print(f"Total trajectories loaded: {len(trajectories)}")
    print(f"Left lane changes: {len(left_delta)}")
    print(f"Right lane changes: {len(right_delta)}")
    print(f"Lane keeping maneuvers: {len(extract_feature_sequences(trajectories, directions, DELTA_Y)[2])}")

    if right_vel:
        right_vel_array = np.array(right_vel)
        boundary_velocities = right_vel_array[:, 100]
        direction_consistency = np.sum(boundary_velocities > 0) / len(boundary_velocities) * 100
        print(f"\nRight LC Lateral Velocity at Boundary:")
        print(f"Direction consistency (positive velocity): {direction_consistency:.1f}%")

    if right_accel:
        right_accel_array = np.array(right_accel)
        peak_accel = np.max(np.abs(right_accel_array), axis=1)
        print(f"\nRight LC Lateral Acceleration Analysis:")
        print(f"Mean peak acceleration magnitude: {np.mean(peak_accel):.2f} m/s²")

    # ADD: Statistics for Longitudinal Velocity  # <-- ADD THIS BLOCK
    if right_x_vel:
        right_x_vel_array = np.array(right_x_vel)
        vel_before = np.mean(right_x_vel_array[:, 50])
        vel_after = np.mean(right_x_vel_array[:, 150])
        print(f"\nRight LC Longitudinal Velocity Analysis:")
        print(f"Mean speed 2s before boundary: {vel_before:.2f} m/s")
        print(f"Mean speed 2s after boundary:  {vel_after:.2f} m/s")
        print(f"Average change in speed: {vel_after - vel_before:+.2f} m/s")

    plt.show()

if __name__ == "__main__":
    plot_trajectories()