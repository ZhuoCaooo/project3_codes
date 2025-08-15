import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def investigate_x_velocity():
    """Investigate X velocity data specifically to find issues"""

    # Load trajectory data
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

    # Extract X velocity (feature index 7)
    X_VELOCITY = 7

    left_x_vel = []
    right_x_vel = []
    lane_keep_x_vel = []

    for traj, dir_seq in zip(trajectories, directions):
        if len(traj) != 200:  # Skip incomplete trajectories
            continue

        x_vel_seq = [frame[X_VELOCITY] for frame in traj]

        if dir_seq[0] == 1:  # Left lane change
            left_x_vel.append(x_vel_seq)
        elif dir_seq[0] == 2:  # Right lane change
            right_x_vel.append(x_vel_seq)
        elif dir_seq[0] == 0:  # Lane keeping
            lane_keep_x_vel.append(x_vel_seq)

    print(f"Left LC: {len(left_x_vel)}, Right LC: {len(right_x_vel)}, Lane Keep: {len(lane_keep_x_vel)}")

    # Investigation plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('X Velocity Investigation', fontsize=16)

    time_axis = np.linspace(-4, 4, 200)

    # Plot 1: Raw X velocity trajectories
    ax1 = axes[0, 0]
    for seq in right_x_vel[:50]:
        ax1.plot(time_axis, seq, 'r-', alpha=0.3, linewidth=0.5)
    for seq in left_x_vel[:50]:
        ax1.plot(time_axis, seq, 'b-', alpha=0.3, linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax1.set_title('Raw X Velocity Trajectories')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Velocity (m/s)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: X velocity distribution at boundary (t=0)
    ax2 = axes[0, 1]
    if right_x_vel:
        right_boundary_vels = [seq[100] for seq in right_x_vel]  # Frame 100 = t=0
        ax2.hist(right_boundary_vels, bins=50, alpha=0.7, color='red', label='Right LC')
    if left_x_vel:
        left_boundary_vels = [seq[100] for seq in left_x_vel]
        ax2.hist(left_boundary_vels, bins=50, alpha=0.7, color='blue', label='Left LC')
    if lane_keep_x_vel:
        keep_boundary_vels = [seq[100] for seq in lane_keep_x_vel]
        ax2.hist(keep_boundary_vels, bins=50, alpha=0.7, color='green', label='Lane Keep')
    ax2.set_title('X Velocity Distribution at Boundary (t=0)')
    ax2.set_xlabel('X Velocity (m/s)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean X velocity with error bars
    ax3 = axes[0, 2]
    if right_x_vel:
        right_array = np.array(right_x_vel)
        mean_right = np.mean(right_array, axis=0)
        std_right = np.std(right_array, axis=0)
        ax3.fill_between(time_axis, mean_right - std_right, mean_right + std_right,
                         alpha=0.3, color='red', label='Right LC ±1σ')
        ax3.plot(time_axis, mean_right, 'r-', linewidth=2, label='Right LC Mean')
    if left_x_vel:
        left_array = np.array(left_x_vel)
        mean_left = np.mean(left_array, axis=0)
        std_left = np.std(left_array, axis=0)
        ax3.fill_between(time_axis, mean_left - std_left, mean_left + std_left,
                         alpha=0.3, color='blue', label='Left LC ±1σ')
        ax3.plot(time_axis, mean_left, 'b-', linewidth=2, label='Left LC Mean')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Boundary')
    ax3.set_title('Mean X Velocity Profile')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('X Velocity (m/s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: X velocity range over time
    ax4 = axes[1, 0]
    all_x_vel = right_x_vel + left_x_vel + lane_keep_x_vel
    if all_x_vel:
        all_array = np.array(all_x_vel)
        min_vals = np.min(all_array, axis=0)
        max_vals = np.max(all_array, axis=0)
        ax4.fill_between(time_axis, min_vals, max_vals, alpha=0.3, color='gray', label='Min-Max Range')
        ax4.plot(time_axis, np.mean(all_array, axis=0), 'k-', linewidth=2, label='Overall Mean')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax4.set_title('X Velocity Range Over Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X Velocity (m/s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Check for outliers - Box plot at different time points
    ax5 = axes[1, 1]
    if all_x_vel:
        all_array = np.array(all_x_vel)
        # Sample at t=-2s, t=0s, t=+2s (frames 50, 100, 150)
        data_points = [all_array[:, 50], all_array[:, 100], all_array[:, 150]]
        ax5.boxplot(data_points, labels=['t=-2s', 't=0s', 't=+2s'])
        ax5.set_title('X Velocity Box Plot at Key Times')
        ax5.set_ylabel('X Velocity (m/s)')
        ax5.grid(True, alpha=0.3)

    # Plot 6: First few raw data samples to inspect
    ax6 = axes[1, 2]
    if right_x_vel:
        for i, seq in enumerate(right_x_vel[:10]):
            ax6.plot(time_axis, seq, '-', alpha=0.8, linewidth=1, label=f'Sample {i + 1}')
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax6.set_title('First 10 Right LC X Velocity Samples')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('X Velocity (m/s)')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print detailed statistics
    print("\n" + "=" * 50)
    print("X VELOCITY INVESTIGATION RESULTS")
    print("=" * 50)

    if right_x_vel:
        right_array = np.array(right_x_vel)
        print(f"\nRIGHT LANE CHANGES ({len(right_x_vel)} samples):")
        print(f"  Range: {np.min(right_array):.2f} to {np.max(right_array):.2f} m/s")
        print(f"  Mean at t=-4s: {np.mean(right_array[:, 0]):.2f} m/s")
        print(f"  Mean at t=0s:  {np.mean(right_array[:, 100]):.2f} m/s")
        print(f"  Mean at t=+4s: {np.mean(right_array[:, -1]):.2f} m/s")
        print(f"  Standard deviation: {np.std(right_array):.2f} m/s")

        # Check for suspicious values
        negative_count = np.sum(right_array < 0)
        total_count = right_array.size
        print(f"  Negative values: {negative_count}/{total_count} ({negative_count / total_count * 100:.1f}%)")

        # Check for extreme values
        q1, q99 = np.percentile(right_array, [1, 99])
        print(f"  1st-99th percentile range: {q1:.2f} to {q99:.2f} m/s")

    if left_x_vel:
        left_array = np.array(left_x_vel)
        print(f"\nLEFT LANE CHANGES ({len(left_x_vel)} samples):")
        print(f"  Range: {np.min(left_array):.2f} to {np.max(left_array):.2f} m/s")
        print(f"  Mean at t=-4s: {np.mean(left_array[:, 0]):.2f} m/s")
        print(f"  Mean at t=0s:  {np.mean(left_array[:, 100]):.2f} m/s")
        print(f"  Mean at t=+4s: {np.mean(left_array[:, -1]):.2f} m/s")

        # Check for suspicious values
        negative_count = np.sum(left_array < 0)
        total_count = left_array.size
        print(f"  Negative values: {negative_count}/{total_count} ({negative_count / total_count * 100:.1f}%)")

    # Check if the coordinate system is the issue
    print(f"\nCOORDINATE SYSTEM ANALYSIS:")
    print(f"From highD documentation:")
    print(f"  - Upper lanes move LEFT (negative X direction)")
    print(f"  - Lower lanes move RIGHT (positive X direction)")
    print(f"  - Your data should reflect this pattern")

    if right_x_vel and left_x_vel:
        right_mean = np.mean(np.array(right_x_vel))
        left_mean = np.mean(np.array(left_x_vel))
        print(f"\nActual means in your data:")
        print(f"  Right LC mean X velocity: {right_mean:.2f} m/s")
        print(f"  Left LC mean X velocity: {left_mean:.2f} m/s")

        if right_mean > 0 and left_mean < 0:
            print("  ✓ This matches expected pattern (right+, left-)")
        elif right_mean < 0 and left_mean > 0:
            print("  ⚠ This is opposite to expected pattern")
        else:
            print("  ⚠ Unexpected pattern - both same sign")

    plt.show()


if __name__ == "__main__":
    investigate_x_velocity()