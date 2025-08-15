import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_velocity_change():
    """
    Calculates and visualizes the change in x-velocity for different maneuvers.
    """
    # --- 1. Load Trajectory Data ---
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

    if not trajectories:
        print("No trajectory data loaded. Exiting.")
        return

    print(f"Loaded {len(trajectories)} total trajectories.")

    # --- 2. Separate Trajectories by Maneuver ---
    X_VELOCITY = 7  # Feature index for x-velocity

    # We only want complete trajectories for consistent analysis
    # A complete trajectory has 200 frames, corresponding to 8 seconds.
    # Frame 100 is t=0s, Frame 199 is t=4s
    left_x_vel = []
    right_x_vel = []
    lane_keep_x_vel = []

    for traj, dir_seq in zip(trajectories, directions):
        if len(traj) == 200:
            x_vel_seq = [frame[X_VELOCITY] for frame in traj]
            if dir_seq[0] == 1:  # Left lane change
                left_x_vel.append(x_vel_seq)
            elif dir_seq[0] == 2:  # Right lane change
                right_x_vel.append(x_vel_seq)
            elif dir_seq[0] == 0:  # Lane keeping
                lane_keep_x_vel.append(x_vel_seq)

    print(f"Found {len(left_x_vel)} Left LCs, {len(right_x_vel)} Right LCs, {len(lane_keep_x_vel)} Lane Keeps.")
    print("-" * 50)

    # --- 3. Calculate Velocity Change (Δv) ---
    # We'll calculate the change from the maneuver point (t=0s) to the end of the clip (t=4s).
    # t=0s corresponds to frame index 100.
    # t=4s corresponds to frame index 199.
    start_frame_idx = 100
    end_frame_idx = 199

    # Calculate delta_v = v_final - v_initial for each trajectory
    left_deltas = [seq[end_frame_idx] - seq[start_frame_idx] for seq in left_x_vel]
    right_deltas = [seq[end_frame_idx] - seq[start_frame_idx] for seq in right_x_vel]
    keep_deltas = [seq[end_frame_idx] - seq[start_frame_idx] for seq in lane_keep_x_vel]

    # --- 4. Print Statistical Summary ---
    print("STATISTICAL SUMMARY OF VELOCITY CHANGE (Δv from t=0s to t=4s)")
    if right_deltas:
        print(f"\nRight Lane Change ({len(right_deltas)} samples):")
        print(f"  Mean Change:   {np.mean(right_deltas):.2f} m/s")
        print(f"  Median Change: {np.median(right_deltas):.2f} m/s")
        print(f"  Std Dev:       {np.std(right_deltas):.2f} m/s")

    if left_deltas:
        print(f"\nLeft Lane Change ({len(left_deltas)} samples):")
        print(f"  Mean Change:   {np.mean(left_deltas):.2f} m/s")
        print(f"  Median Change: {np.median(left_deltas):.2f} m/s")
        print(f"  Std Dev:       {np.std(left_deltas):.2f} m/s")

    if keep_deltas:
        print(f"\nLane Keep ({len(keep_deltas)} samples):")
        print(f"  Mean Change:   {np.mean(keep_deltas):.2f} m/s")
        print(f"  Median Change: {np.median(keep_deltas):.2f} m/s")
        print(f"  Std Dev:       {np.std(keep_deltas):.2f} m/s")
    print("-" * 50)

    # --- 5. Visualize the Distribution of Velocity Changes ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Create histograms to show the distribution of velocity changes
    # A positive value means acceleration, a negative value means deceleration.
    bins = np.linspace(-5, 5, 60)  # Define shared bins for comparable histograms

    plt.hist(right_deltas, bins=bins, alpha=0.7, color='red', label=f'Right LC (Mean: {np.mean(right_deltas):.2f})')
    plt.hist(left_deltas, bins=bins, alpha=0.7, color='blue', label=f'Left LC (Mean: {np.mean(left_deltas):.2f})')
    plt.hist(keep_deltas, bins=bins, alpha=0.6, color='green', label=f'Lane Keep (Mean: {np.mean(keep_deltas):.2f})')

    # Add a vertical line at 0 to clearly distinguish acceleration from deceleration
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No Speed Change')

    plt.title('Distribution of X-Velocity Change During Maneuver (t=0s to t=4s)', fontsize=16)
    plt.xlabel('Velocity Change (Δv) in m/s\n(Positive = Acceleration, Negative = Deceleration)', fontsize=12)
    plt.ylabel('Number of Trajectories', fontsize=12)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    analyze_velocity_change()