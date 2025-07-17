#!/usr/bin/env python3
"""
Convert numerical trajectory data to LC-LLM format (matching actual paper format)
FIXED: Properly handles 25 Hz data structure (150 frames total)
- 4s before crossing = 100 frames (frames 0-99)
- 2s after crossing = 50 frames (frames 100-149)
- Current frame = frame 99 (just before crossing)
- Crossing boundary = frame 100
"""

import json
import numpy as np
from typing import List, Tuple, Dict


class LCLLMDataConverter:
    def __init__(self):
        # Feature indices from your construct_features function
        self.LEFT_LANE_EXIST = 0
        self.RIGHT_LANE_EXIST = 1
        self.DELTA_Y = 2
        self.Y_VELOCITY = 3
        self.Y_ACCELERATION = 4
        self.X_VELOCITY = 5
        self.X_ACCELERATION = 6
        self.CAR_TYPE = 7
        self.PRECEDING_TTC = 8
        self.FOLLOWING_TTC = 9
        self.LEFT_PRECEDING_TTC = 10
        self.LEFT_ALONGSIDE_TTC = 11
        self.LEFT_FOLLOWING_TTC = 12
        self.RIGHT_PRECEDING_TTC = 13
        self.RIGHT_ALONGSIDE_TTC = 14
        self.RIGHT_FOLLOWING_TTC = 15

        # System message (exact format from paper)
        self.SYSTEM_MESSAGE = """Role: You are an expert driving prediction model of an autonomous driving system, that can predict the future driving intention and future 4-second driving trajectory for a given target vehicle, avoiding collision with other vehicles and obstacles on the road.
Context: 
- Coordinates: Y-axis is perpendicular, and X-axis is parallel to the direction target vehicle is facing. target vehicle's current position is (0,0). Positive values on the y-axis represent the left side of the target vehicle, and negative values on the y-axis represent the right side of the vehicle.
Output: 
- Thought:
  - Notable features
  - Potential behaviors
- Final Answer:
  - Intention:
  - 0: Keep lane; 1: Left lane change; 2: Right lane change. The final answer should be one of the three modes.
  - Trajectory (MOST IMPORTANT): 4 points, one every 1 second
  - [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]"""

    def determine_lane_configuration(self, current_frame: Tuple) -> Tuple[str, int]:
        """Determine lane position and total lanes"""
        left_exists = current_frame[self.LEFT_LANE_EXIST] > 0.5
        right_exists = current_frame[self.RIGHT_LANE_EXIST] > 0.5

        if left_exists and right_exists:
            return "middle", 3  # At least 3 lanes, vehicle in middle
        elif left_exists and not right_exists:
            return "rightmost", 2  # Vehicle in rightmost lane
        elif not left_exists and right_exists:
            return "leftmost", 2  # Vehicle in leftmost lane
        else:
            return "single", 1  # Single lane (rare)

    def create_historical_positions(self, history_frames: List[Tuple]) -> str:
        """Create 6 historical positions properly for 25 Hz data"""
        positions = []

        # For 25 Hz data, sample every 10 frames (0.4s intervals) from last 2 seconds (50 frames)
        # We want 6 positions going back 2 seconds at 0.4s intervals
        sample_interval = 10  # 0.4s * 25 Hz = 10 frames

        if len(history_frames) < 60:  # Need at least 60 frames for 6 positions
            # Pad with the first frame if not enough history
            padded_frames = [history_frames[0]] * (60 - len(history_frames)) + history_frames
        else:
            # Take last 60 frames (2.4 seconds)
            padded_frames = history_frames[-60:]

        # Current frame (time 0) should be at (0, 0)
        current_frame = padded_frames[-1]  # Last frame is current
        current_y = current_frame[self.DELTA_Y]

        # Sample every 10th frame, going backwards
        for i in range(6):
            frame_idx = len(padded_frames) - 1 - (i * sample_interval)
            frame_idx = max(0, frame_idx)  # Ensure valid index
            frame = padded_frames[frame_idx]

            time_offset = i * 0.4  # 0.4s intervals, going backwards
            x_velocity = frame[self.X_VELOCITY]  # KEEP sign for direction
            x_pos = -time_offset * x_velocity  # Negative offset (going back in time)

            # FIXED: Calculate y position relative to current frame being (0,0)
            if i == 0:  # Current frame
                y_pos = 0.0  # Current position is always (0,0)
            else:
                # Calculate relative y position from current frame
                y_pos = frame[self.DELTA_Y] - current_y

            positions.append(f"({x_pos:.2f},{y_pos:.2f})")

        # Reverse to get chronological order (oldest to newest)
        positions.reverse()
        return ", ".join(positions)

    def create_surrounding_vehicles_info(self, current_frame: Tuple) -> List[str]:
        """Better surrounding vehicle info"""
        surrounding_info = []

        # Use speed magnitude for comparisons, but preserve direction in descriptions
        ego_speed = abs(current_frame[self.X_VELOCITY]) * 3.6  # km/h magnitude
        ego_velocity_kmh = current_frame[self.X_VELOCITY] * 3.6  # With direction

        # Ahead vehicle
        if current_frame[self.PRECEDING_TTC] < 200:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            # Estimate speed with direction preserved
            if current_frame[self.PRECEDING_TTC] < 50:
                speed = ego_velocity_kmh - (15 * (1 if ego_velocity_kmh > 0 else -1))
            else:
                speed = ego_velocity_kmh + (5 * (1 if ego_velocity_kmh > 0 else -1))
            surrounding_info.append(
                f"- Ahead: a {vehicle_type} traveling at {speed:.2f} km/h of X-axis, "
                f"with a distance of {current_frame[self.PRECEDING_TTC]:.0f} m."
            )

        # Behind vehicle
        if current_frame[self.FOLLOWING_TTC] < 200:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            speed = ego_velocity_kmh - (10 * (1 if ego_velocity_kmh > 0 else -1))
            surrounding_info.append(
                f"- Behind: a {vehicle_type} traveling at {speed:.2f} km/h of X-axis, "
                f"with a distance of {current_frame[self.FOLLOWING_TTC]:.0f} m."
            )

        # Left front
        if current_frame[self.LEFT_PRECEDING_TTC] < 200:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            speed = ego_velocity_kmh + (15 * (1 if ego_velocity_kmh > 0 else -1))
            surrounding_info.append(
                f"- Left front: a {vehicle_type} traveling at {speed:.2f} km/h of X-axis, "
                f"with a distance of {current_frame[self.LEFT_PRECEDING_TTC]:.0f} m."
            )

        # Add more vehicles for realism...

        return surrounding_info

    def generate_notable_features(self, current_frame: Tuple, lane_position: str) -> List[str]:
        """Better logic for vehicle analysis"""
        features = []

        # Fix velocity handling - use absolute values for speed comparison
        ego_speed = abs(current_frame[self.X_VELOCITY]) * 3.6  # km/h

        # Lateral velocity (if significant)
        vy_kmh = current_frame[self.Y_VELOCITY] * 3.6
        if abs(vy_kmh) > 1.0:
            features.append(f"Notable features: vy = {vy_kmh:.2f}")

        # Acceleration
        ax = current_frame[self.X_ACCELERATION]
        if abs(ax) > 0.5:
            features.append(f"Notable features: ax = {ax:.2f}")

        # Ahead vehicle analysis
        if current_frame[self.PRECEDING_TTC] < 200:
            if current_frame[self.PRECEDING_TTC] < 80:  # Close vehicle
                features.append("Notable feature: Ahead is block.")
            else:
                features.append("Notable feature: Ahead is free.")
        else:
            features.append("Notable feature: Ahead is free.")

        # Left lane analysis with proper speed comparison
        if current_frame[self.LEFT_PRECEDING_TTC] < 200:
            # Assume left vehicle speed (estimate based on distance and lane)
            left_distance = current_frame[self.LEFT_PRECEDING_TTC]
            if left_distance > 50:  # Safe gap
                features.append("Notable feature: Left front is free.")
            else:
                features.append("Notable feature: Left is block.")
        else:
            features.append("Notable feature: Left front is free.")

        # Right lane analysis
        if current_frame[self.RIGHT_LANE_EXIST] > 0.5:
            if current_frame[self.RIGHT_PRECEDING_TTC] < 200:
                if current_frame[self.RIGHT_PRECEDING_TTC] > 50:
                    features.append("Notable feature: Right front is free.")
                else:
                    features.append("Notable feature: Right is block.")

        # Lane position
        features.append(f"Notable feature: {lane_position} lane")

        return features

    def determine_potential_behavior(self, current_frame: Tuple, intention: int,
                                     lane_position: str) -> str:
        """Determine potential behavior based on context"""

        if intention == 0:  # Keep lane
            if current_frame[self.PRECEDING_TTC] < 50:
                return "Following and keep lane."
            else:
                return "Normal keep lane."

        elif intention == 1:  # Left lane change
            if (current_frame[self.PRECEDING_TTC] < 100 and
                    lane_position in ["rightmost", "middle"]):
                return "Change to the left lane for overtaking"
            elif current_frame[self.X_ACCELERATION] > 1.0:
                return "Change left to the fast lane."
            else:
                return "Irregular left lane change."

        else:  # intention == 2, Right lane change
            if (current_frame[self.PRECEDING_TTC] < 100 and
                    lane_position in ["leftmost", "middle"]):
                return "Change to the right lane for overtaking."
            elif (current_frame[self.X_ACCELERATION] < -0.5 or
                  current_frame[self.CAR_TYPE] < 0):  # Deceleration or truck
                return "Change right to the slow lane."
            else:
                return "Irregular right lane change."

    def extract_actual_future_trajectory(self, current_frame: Tuple, future_frames: List[Tuple]) -> str:
        """Extract actual future trajectory from the dataset (2 seconds after crossing)"""
        trajectory_points = []

        # Current position is (0,0)
        current_y = 0.0
        current_x = 0.0

        # Extract trajectory from actual future data (2 seconds = 50 frames at 25 Hz)
        # Sample at appropriate intervals to get trajectory points

        if len(future_frames) >= 50:  # We have full 2 seconds of future data
            # Sample every 25 frames (1 second intervals) for 2 points
            # Or sample every 12.5 frames (0.5 second intervals) for 4 points
            # Let's use 4 points at 0.5 second intervals to match LC-LLM format

            sample_indices = [12, 25, 37, 49]  # ~0.5s, 1s, 1.5s, 2s intervals

            for i, frame_idx in enumerate(sample_indices):
                if frame_idx < len(future_frames):
                    future_frame = future_frames[frame_idx]

                    # Calculate actual time step
                    time_step = (frame_idx + 1) / 25.0  # Convert frame to seconds

                    # X position: use actual velocity and time
                    vx = current_frame[self.X_VELOCITY]
                    x_pos = vx * time_step

                    # Y position: extract from actual future data, relative to current (0,0)
                    future_y = future_frame[self.DELTA_Y]
                    y_pos = future_y - current_frame[self.DELTA_Y]  # Relative to current frame

                    trajectory_points.append(f"({x_pos:.2f},{y_pos:.2f})")
                else:
                    # Fallback if not enough future data
                    break

        # If we don't have enough points, pad with the last known trajectory
        while len(trajectory_points) < 4:
            if trajectory_points:
                # Extend the last point
                last_point = trajectory_points[-1]
                trajectory_points.append(last_point)
            else:
                # Fallback to simple calculation
                time_step = len(trajectory_points) + 1
                vx = current_frame[self.X_VELOCITY]
                x_pos = vx * time_step * 0.5  # 0.5 second intervals
                y_pos = 0.0  # No movement if no data
                trajectory_points.append(f"({x_pos:.2f},{y_pos:.2f})")

        return "[" + ", ".join(trajectory_points[:4]) + "]"

    def convert_sample_to_lcllm_format(self, features_sequence: List[Tuple],
                                       direction_labels: List[int]) -> Dict:
        """FIXED: Convert trajectory sample to exact LC-LLM format with proper temporal structure"""

        # From your extraction script: 25 Hz data, 150 frames total (6 seconds)
        # 4s before crossing = 100 frames (0-99), 2s after crossing = 50 frames (100-149)
        # Crossing boundary is ALWAYS at frame 100
        crossing_idx = 100  # Always at 4 seconds with 25 Hz data

        # Current frame is just before crossing (end of 4s before period)
        current_idx = crossing_idx - 1  # Frame 99 (last frame before crossing)
        current_frame = features_sequence[current_idx]

        # History is the 4s before crossing (frames 0-99, 100 frames total)
        history_frames = features_sequence[:crossing_idx]

        # Future is the 2s after crossing (frames 100-149, this is what we want to predict)
        future_frames = features_sequence[crossing_idx:]

        # Intention should be determined from the future period (what WILL happen)
        future_labels = direction_labels[crossing_idx:]
        if future_labels:
            intention = max(set(future_labels), key=future_labels.count)
        else:
            # Fallback: use the first frame after crossing
            intention = direction_labels[crossing_idx] if crossing_idx < len(direction_labels) else 0

        # Generate scenario components (using only information available before crossing)
        lane_position, lane_count = self.determine_lane_configuration(current_frame)

        # Vehicle information
        car_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
        vx_kmh = current_frame[self.X_VELOCITY] * 3.6
        vy_kmh = current_frame[self.Y_VELOCITY] * 3.6
        ax = current_frame[self.X_ACCELERATION]
        ay = current_frame[self.Y_ACCELERATION]

        # Vehicle dimensions (estimates based on type)
        if car_type == "Car":
            width, length = np.random.uniform(1.8, 2.2), np.random.uniform(4.0, 5.5)
        else:
            width, length = 2.5, np.random.uniform(12.0, 22.0)

        # Historical positions (using only pre-crossing data)
        historical_positions = self.create_historical_positions(history_frames)

        # Surrounding vehicles (using only pre-crossing data)
        surrounding_vehicles = self.create_surrounding_vehicles_info(current_frame)

        # Create scenario description
        lane_description = f"a {lane_count}-lane highway" if lane_count > 1 else "highway"
        scenario = f"""The target vehicle is driving on {lane_description}, located at the {lane_position} lane.
The information of target vehicle is as follow:
  - Velocity(km/h): vx={vx_kmh:.2f}, vy={vy_kmh:.2f}
  - Accelaration: ax={ax:.2f}, ay={ay:.2f}
  - Type: {car_type}, with width of {width:.2f} m and length of {length:.2f} m
  - Historical position of the last 2 seconds (One point every 0.4s): [{historical_positions}]

The information of its surrounding vehicles (with a range of 200m) are listed as follow:
  {chr(10).join(surrounding_vehicles) if surrounding_vehicles else "  - "}"""

        # Generate reasoning (using only pre-crossing information)
        notable_features = self.generate_notable_features(current_frame, lane_position)
        potential_behavior = self.determine_potential_behavior(current_frame, intention, lane_position)

        # FIXED: Extract actual trajectory from future data (2 seconds after crossing)
        future_trajectory = self.extract_actual_future_trajectory(current_frame, future_frames)

        # Map intention
        intention_map = {0: "0: Keep lane", 1: "1: Left lane change", 2: "2: Right lane change"}
        intention_text = intention_map[intention]

        # Create reasoning and response
        thought_section = "Thought:\n  - " + "\n  - ".join(notable_features)
        thought_section += f"\n  - Potential behavior: {potential_behavior}."

        response = f"""{thought_section}
Final Answer:
  - Intention: \"{intention_text}\"
  - Trajectory: \"{future_trajectory}\"
 """

        # Create final Llama format
        llama_sample = {
            "text": f"<s>[INST] <<SYS>>\n{self.SYSTEM_MESSAGE}\n<</SYS>>\n\n{scenario} [/INST] {response}</s>"
        }

        return llama_sample


def main():
    """Convert your pickle data to LC-LLM format - generates both train and test data"""
    import pickle
    import glob

    converter = LCLLMDataConverter()

    # Process your data - expecting 6s total at 25 Hz: 4s before + 2s after crossing
    # 25 Hz = 150 frames total (100 before + 50 after)
    pickle_files = glob.glob("output_4sbefore_2safter/*.pickle")

    print(f"Found {len(pickle_files)} pickle files total")

    # Generate training data (files 1-50)
    print("\n" + "=" * 50)
    print("GENERATING TRAINING DATA")
    print("=" * 50)
    train_files = pickle_files[0:50]  # Files 1-50
    train_samples = []

    for file_path in train_files:
        print(f"Processing {file_path}...")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for features_sequence, direction_labels in data:
            if len(features_sequence) == 150:  # Exactly 150 frames: 4s before + 2s after crossing at 25 Hz
                try:
                    sample = converter.convert_sample_to_lcllm_format(
                        features_sequence, direction_labels
                    )
                    train_samples.append(sample)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue

    # Save training data
    with open("lcllm_training_data.json", 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"✓ Saved {len(train_samples)} TRAINING samples to lcllm_training_data.json")

    # Generate testing data (files 51-60)
    print("\n" + "=" * 50)
    print("GENERATING TESTING DATA")
    print("=" * 50)
    test_files = pickle_files[50:60]  # Files 51-60
    test_samples = []

    for file_path in test_files:
        print(f"Processing {file_path}...")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for features_sequence, direction_labels in data:
            if len(features_sequence) == 150:  # Exactly 150 frames: 4s before + 2s after crossing at 25 Hz
                try:
                    sample = converter.convert_sample_to_lcllm_format(
                        features_sequence, direction_labels
                    )
                    test_samples.append(sample)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue

    # Save testing data
    with open("lcllm_testing_data.json", 'w') as f:
        json.dump(test_samples, f, indent=2)
    print(f"✓ Saved {len(test_samples)} TESTING samples to lcllm_testing_data.json")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Training files processed: {len(train_files)} (files 1-50)")
    print(f"Testing files processed: {len(test_files)} (files 51-60)")
    print(f"Training samples: {len(train_samples)}")
    print(f"Testing samples: {len(test_samples)}")
    print(f"Total samples: {len(train_samples) + len(test_samples)}")

    # Show example
    if train_samples:
        print("\n" + "=" * 70)
        print("EXAMPLE OUTPUT (matches paper format):")
        print("=" * 70)
        example_text = train_samples[0]["text"]
        # Pretty print by splitting at key sections
        parts = example_text.split("[/INST]")
        print("INPUT:", parts[0].replace("<s>[INST]", "").strip()[:500] + "...")
        print("\nOUTPUT:", parts[1].replace("</s>", "").strip()[:300] + "...")


if __name__ == "__main__":
    main()