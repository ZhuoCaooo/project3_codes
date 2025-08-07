import pickle
import os
import numpy as np


def test_coordinate_system():
    """
    Test script to confirm coordinate system:
    - Positive Y = left side of vehicle
    - Negative Y = right side of vehicle
    - X = forward direction (should generally increase)
    """

    # Feature indices based on your code
    X_POS_IDX = 5
    Y_POS_IDX = 6
    X_VEL_IDX = 7
    Y_VEL_IDX = 3

    total_trajectories = 0
    left_changes = 0
    right_changes = 0
    lane_keeping = 0

    print("Testing coordinate system...")
    print("=" * 50)

    # Test a few files to get a sample
    for file_num in [1, 5, 10, 15, 20]:
        filename = f"output_4sbefore_4safter/result{file_num:02d}.pickle"

        if not os.path.exists(filename):
            continue

        print(f"\nTesting file: {filename}")

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        for traj_idx, (trajectory, directions) in enumerate(data):
            if len(trajectory) < 50:  # Skip very short trajectories
                continue

            total_trajectories += 1
            direction = directions[0]  # Get the direction (0=LK, 1=Left, 2=Right)

            # Extract positions
            x_positions = [frame[X_POS_IDX] for frame in trajectory]
            y_positions = [frame[Y_POS_IDX] for frame in trajectory]
            x_velocities = [frame[X_VEL_IDX] for frame in trajectory]
            y_velocities = [frame[Y_VEL_IDX] for frame in trajectory]

            # Calculate changes
            x_change = x_positions[-1] - x_positions[0]
            y_change = y_positions[-1] - y_positions[0]
            avg_x_vel = np.mean(x_velocities)
            avg_y_vel = np.mean(y_velocities)

            if direction == 1:  # Left lane change
                left_changes += 1
                if left_changes <= 3:  # Show first 3 examples
                    print(f"\nLEFT LANE CHANGE #{left_changes}:")
                    print(f"  X change: {x_change:.2f} (should be positive - forward motion)")
                    print(f"  Y change: {y_change:.2f} (should be positive - moving left)")
                    print(f"  Avg X velocity: {avg_x_vel:.2f}")
                    print(f"  Avg Y velocity: {avg_y_vel:.2f}")
                    print(f"  Start pos: ({x_positions[0]:.2f}, {y_positions[0]:.2f})")
                    print(f"  End pos: ({x_positions[-1]:.2f}, {y_positions[-1]:.2f})")

            elif direction == 2:  # Right lane change
                right_changes += 1
                if right_changes <= 3:  # Show first 3 examples
                    print(f"\nRIGHT LANE CHANGE #{right_changes}:")
                    print(f"  X change: {x_change:.2f} (should be positive - forward motion)")
                    print(f"  Y change: {y_change:.2f} (should be negative - moving right)")
                    print(f"  Avg X velocity: {avg_x_vel:.2f}")
                    print(f"  Avg Y velocity: {avg_y_vel:.2f}")
                    print(f"  Start pos: ({x_positions[0]:.2f}, {y_positions[0]:.2f})")
                    print(f"  End pos: ({x_positions[-1]:.2f}, {y_positions[-1]:.2f})")

            elif direction == 0:  # Lane keeping
                lane_keeping += 1
                if lane_keeping <= 2:  # Show first 2 examples
                    print(f"\nLANE KEEPING #{lane_keeping}:")
                    print(f"  X change: {x_change:.2f} (should be positive - forward motion)")
                    print(f"  Y change: {y_change:.2f} (should be close to 0 - staying in lane)")
                    print(f"  Avg X velocity: {avg_x_vel:.2f}")
                    print(f"  Avg Y velocity: {avg_y_vel:.2f}")

            # Stop after showing enough examples
            if left_changes >= 3 and right_changes >= 3 and lane_keeping >= 2:
                break

        if left_changes >= 3 and right_changes >= 3 and lane_keeping >= 2:
            break

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total trajectories tested: {total_trajectories}")
    print(f"Left lane changes: {left_changes}")
    print(f"Right lane changes: {right_changes}")
    print(f"Lane keeping: {lane_keeping}")

    print("\nEXPECTED COORDINATE SYSTEM:")
    print("- X should increase (forward motion)")
    print("- Left lane changes: Y should increase (positive = left)")
    print("- Right lane changes: Y should decrease (negative = right)")
    print("- Lane keeping: Y should stay relatively constant")


def detailed_trajectory_analysis():
    """
    More detailed analysis of a single trajectory
    """
    print("\n" + "=" * 60)
    print("DETAILED TRAJECTORY ANALYSIS")
    print("=" * 60)

    X_POS_IDX = 5
    Y_POS_IDX = 6

    # Load first available file
    for file_num in range(1, 61):
        filename = f"output_4sbefore_4safter/result{file_num:02d}.pickle"
        if os.path.exists(filename):
            break
    else:
        print("No trajectory files found!")
        return

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # Find a left lane change trajectory
    for trajectory, directions in data:
        if directions[0] == 1 and len(trajectory) >= 100:  # Left change, good length
            print(f"\nDetailed LEFT LANE CHANGE trajectory:")
            print("Frame | X Position | Y Position | X Change | Y Change")
            print("-" * 55)

            for i in range(0, len(trajectory), 25):  # Show every 25th frame
                x_pos = trajectory[i][X_POS_IDX]
                y_pos = trajectory[i][Y_POS_IDX]

                if i == 0:
                    x_change = 0
                    y_change = 0
                else:
                    x_change = x_pos - trajectory[0][X_POS_IDX]
                    y_change = y_pos - trajectory[0][Y_POS_IDX]

                print(f"{i:5d} | {x_pos:10.2f} | {y_pos:10.2f} | {x_change:8.2f} | {y_change:8.2f}")
            break

    print("\nExpected pattern:")
    print("- X Change should continuously increase (forward motion)")
    print("- Y Change should increase over time (moving left)")


if __name__ == "__main__":
    # Check if output directory exists
    if not os.path.exists("output_4sbefore_4safter"):
        print("Error: output_4sbefore_4safter directory not found!")
        print("Please make sure you have run the trajectory extraction first.")
        exit(1)

    # Run the tests
    test_coordinate_system()
    detailed_trajectory_analysis()