import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def investigate_x_acceleration():
    """
    Investigates X-axis acceleration data from vehicle trajectories to identify
    patterns, outliers, and potential issues.
    """

    # --- 1. Data Loading ---
    trajectories = []
    directions = []

    folder = Path("output_4sbefore_4safter")
    if not folder.exists():
        print(f"Error: Directory not found -> {folder.resolve()}")
        return

    for i in range(1, 61):
        filename = folder / f"result{i:02d}.pickle"
        if filename.exists():
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                for trajectory, direction in data:
                    trajectories.append(trajectory)
                    directions.append(direction)

    print(f"Loaded {len(trajectories)} trajectories")

    # --- 2. Feature Extraction ---
    # The feature index for X acceleration is 8
    X_ACCELERATION = 8

    left_x_accel = []
    right_x_accel = []
    lane_keep_x_accel = []

    for traj, dir_seq in zip(trajectories, directions):
        if len(traj) != 200:  # Skip incomplete trajectories
            continue

        # Extract the acceleration sequence for each trajectory
        x_accel_seq = [frame[X_ACCELERATION] for frame in traj]

        if dir_seq[0] == 1:  # Left lane change
            left_x_accel.append(x_accel_seq)
        elif dir_seq[0] == 2:  # Right lane change
            right_x_accel.append(x_accel_seq)
        elif dir_seq[0] == 0:  # Lane keeping
            lane_keep_x_accel.append(x_accel_seq)

    print(f"Left LC: {len(left_x_accel)}, Right LC: {len(right_x_accel)}, Lane Keep: {len(lane_keep_x_accel)}")

    # --- 3. Investigation Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('X Acceleration Investigation', fontsize=16)

    time_axis = np.linspace(-4, 4, 200)

    # Plot 1: Raw X acceleration trajectories
    ax1 = axes[0, 0]
    for seq in right_x_accel[:50]:
        ax1.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_x_accel[:50]:
        ax1.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)  # Zero-acceleration line
    ax1.set_title('Raw X Acceleration Trajectories')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Acceleration (m/s²)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: X acceleration distribution at boundary (t=0)
    ax2 = axes[0, 1]
    if right_x_accel:
        right_boundary_accels = [seq[100] for seq in right_x_accel]
        ax2.hist(right_boundary_accels, bins=50, alpha=0.7, color='red', label='Right LC')
    if left_x_accel:
        left_boundary_accels = [seq[100] for seq in left_x_accel]
        ax2.hist(left_boundary_accels, bins=50, alpha=0.7, color='blue', label='Left LC')
    if lane_keep_x_accel:
        keep_boundary_accels = [seq[100] for seq in lane_keep_x_accel]
        ax2.hist(keep_boundary_accels, bins=50, alpha=0.7, color='green', label='Lane Keep')
    ax2.set_title('X Acceleration Distribution at Boundary (t=0)')
    ax2.set_xlabel('X Acceleration (m/s²)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean X acceleration with error bars
    ax3 = axes[0, 2]
    if right_x_accel:
        right_array = np.array(right_x_accel)
        mean_right = np.mean(right_array, axis=0)
        std_right = np.std(right_array, axis=0)
        ax3.fill_between(time_axis, mean_right - std_right, mean_right + std_right,
                         alpha=0.3, color='red', label='Right LC ±1σ')
        ax3.plot(time_axis, mean_right, 'r-', linewidth=2, label='Right LC Mean')
    if left_x_accel:
        left_array = np.array(left_x_accel)
        mean_left = np.mean(left_array, axis=0)
        std_left = np.std(left_array, axis=0)
        ax3.fill_between(time_axis, mean_left - std_left, mean_left + std_left,
                         alpha=0.3, color='blue', label='Left LC ±1σ')
        ax3.plot(time_axis, mean_left, 'b-', linewidth=2, label='Left LC Mean')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Boundary')
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_title('Mean X Acceleration Profile')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('X Acceleration (m/s²)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: X acceleration range over time
    ax4 = axes[1, 0]
    all_x_accel = right_x_accel + left_x_accel + lane_keep_x_accel
    if all_x_accel:
        all_array = np.array(all_x_accel)
        min_vals = np.min(all_array, axis=0)
        max_vals = np.max(all_array, axis=0)
        ax4.fill_between(time_axis, min_vals, max_vals, alpha=0.3, color='gray', label='Min-Max Range')
        ax4.plot(time_axis, np.mean(all_array, axis=0), 'k-', linewidth=2, label='Overall Mean')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_title('X Acceleration Range Over Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X Acceleration (m/s²)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Check for outliers - Box plot at different time points
    ax5 = axes[1, 1]
    if all_x_accel:
        all_array = np.array(all_x_accel)
        # Sample at t=-2s, t=0s, t=+2s (frames 50, 100, 150)
        data_points = [all_array[:, 50], all_array[:, 100], all_array[:, 150]]
        ax5.boxplot(data_points, labels=['t=-2s', 't=0s', 't=+2s'])
        ax5.set_title('X Acceleration Box Plot at Key Times')
        ax5.set_ylabel('X Acceleration (m/s²)')
        ax5.grid(True, alpha=0.3)

    # Plot 6: First few raw data samples to inspect
    ax6 = axes[1, 2]
    if right_x_accel:
        for i, seq in enumerate(right_x_accel[:10]):
            ax6.plot(time_axis, seq, '-', alpha=0.8, linewidth=1, label=f'Sample {i + 1}')
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax6.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax6.set_title('First 10 Right LC X Accel Samples')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('X Acceleration (m/s²)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 4. Statistical Analysis ---
    print("\n" + "=" * 50)
    print("X ACCELERATION INVESTIGATION RESULTS")
    print("=" * 50)

    if right_x_accel:
        right_array = np.array(right_x_accel)
        print(f"\nRIGHT LANE CHANGES ({len(right_x_accel)} samples):")
        print(f"  Overall Range: {np.min(right_array):.2f} to {np.max(right_array):.2f} m/s²")
        print(f"  Mean at t=-4s: {np.mean(right_array[:, 0]):.2f} m/s²")
        print(f"  Mean at t=0s:  {np.mean(right_array[:, 100]):.2f} m/s²")
        print(f"  Mean at t=+4s: {np.mean(right_array[:, -1]):.2f} m/s²")
        print(f"  Overall Standard deviation: {np.std(right_array):.2f} m/s²")
        q1, q99 = np.percentile(right_array, [1, 99])
        print(f"  1st-99th percentile range: {q1:.2f} to {q99:.2f} m/s²")

    if left_x_accel:
        left_array = np.array(left_x_accel)
        print(f"\nLEFT LANE CHANGES ({len(left_x_accel)} samples):")
        print(f"  Overall Range: {np.min(left_array):.2f} to {np.max(left_array):.2f} m/s²")
        print(f"  Mean at t=-4s: {np.mean(left_array[:, 0]):.2f} m/s²")
        print(f"  Mean at t=0s:  {np.mean(left_array[:, 100]):.2f} m/s²")
        print(f"  Mean at t=+4s: {np.mean(left_array[:, -1]):.2f} m/s²")
        print(f"  Overall Standard deviation: {np.std(left_array):.2f} m/s²")
        q1, q99 = np.percentile(left_array, [1, 99])
        print(f"  1st-99th percentile range: {q1:.2f} to {q99:.2f} m/s²")

    print("\n" + "=" * 50)
    print("ACCELERATION BEHAVIOR ANALYSIS")
    print("=" * 50)
    print("For acceleration, values are expected to be centered around 0 m/s².")
    print("Significant deviations from zero in the mean profile can indicate")
    print("systematic speeding up or slowing down during maneuvers.")

    if all_x_accel:
        all_array = np.array(all_x_accel)
        overall_mean = np.mean(all_array)
        overall_std = np.std(all_array)
        print(f"\nOverall Mean Acceleration: {overall_mean:.3f} m/s²")
        print(f"Overall Std Dev: {overall_std:.3f} m/s²")

        # Check mean acceleration during the maneuver (e.g., -1s to +1s)
        maneuver_slice = all_array[:, 75:125]  # Frames for t = -1s to +1s
        mean_during_maneuver = np.mean(maneuver_slice)
        print(f"Mean Acceleration during maneuver (-1s to +1s): {mean_during_maneuver:.3f} m/s²")

    plt.show()


if __name__ == "__main__":
    investigate_x_acceleration()