#!/usr/bin/env python3
"""
Fixed LC-LLM Data Converter - HighD Dataset Aware
FIXES:
1. Handles HighD coordinate system (top lanes going left, bottom lanes going right)
2. Corrects velocity interpretation (accounts for sign flips)
3. Uses proper reference frames for position calculation
4. Handles realistic highway driving scenarios
5. ✅ CORRECTED FEATURE INDICES to match updated data processing script
"""

import json
import numpy as np
from typing import List, Tuple, Dict


class FixedLCLLMDataConverter:
    def __init__(self):
        # ✅ CORRECTED Feature indices to match the updated HighD processing script
        self.LEFT_LANE_EXIST = 0
        self.RIGHT_LANE_EXIST = 1
        self.DELTA_Y = 2
        self.Y_VELOCITY = 3        # ✅ CORRECTED: was 5, now 3
        self.Y_ACCELERATION = 4    # ✅ CORRECTED: was 6, now 4
        self.X_POSITION = 5        # ✅ CORRECTED: was 3, now 5
        self.Y_POSITION = 6        # ✅ CORRECTED: was 4, now 6
        self.X_VELOCITY = 7
        self.X_ACCELERATION = 8
        self.CAR_TYPE = 9
        self.PRECEDING_DISTANCE = 10
        self.FOLLOWING_DISTANCE = 11
        self.LEFT_PRECEDING_DISTANCE = 12
        self.LEFT_ALONGSIDE_DISTANCE = 13
        self.LEFT_FOLLOWING_DISTANCE = 14
        self.RIGHT_PRECEDING_DISTANCE = 15
        self.RIGHT_ALONGSIDE_DISTANCE = 16
        self.RIGHT_FOLLOWING_DISTANCE = 17

        # Around line 45, update this part of SYSTEM_MESSAGE:
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
          - Trajectory (MOST IMPORTANT): 20 points, one every 0.2 second
          - [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6), (x7,y7), (x8,y8), (x9,y9), (x10,y10), (x11,y11), (x12,y12), (x13,y13), (x14,y14), (x15,y15), (x16,y16), (x17,y17), (x18,y18), (x19,y19), (x20,y20)]"""

    def determine_traffic_direction(self, features_sequence: List[Tuple]) -> int:
        """Determine traffic direction from X velocity pattern"""
        # Sample a few frames to determine overall direction
        sample_frames = features_sequence[::10]  # Every 10th frame
        x_velocities = [frame[self.X_VELOCITY] for frame in sample_frames]
        avg_x_velocity = np.mean(x_velocities)

        # In HighD:
        # going = 1 (top lanes): moving leftward, X velocity should be negative
        # going = 2 (bottom lanes): moving rightward, X velocity should be positive
        if avg_x_velocity < 0:
            return 1  # Top lanes, going left
        else:
            return 2  # Bottom lanes, going right

    def get_corrected_velocities(self, frame: Tuple, traffic_direction: int) -> Tuple[float, float]:
        """Get corrected velocities accounting for HighD coordinate system"""
        x_vel = frame[self.X_VELOCITY]  # m/s
        y_vel = frame[self.Y_VELOCITY]  # m/s (potentially sign-flipped)

        # The Y velocity was negated for top lanes in your processing script
        # We need to understand what this means for lane change interpretation

        # For LC-LLM format, we want:
        # - Positive Y velocity = moving toward left lane
        # - Negative Y velocity = moving toward right lane

        if traffic_direction == 1:  # Top lanes (going left)
            # Y velocity was negated, so we need to interpret it correctly
            # In HighD top lanes: positive Y = moving toward bottom of screen
            # For lane changes: positive Y should mean moving to left lane
            corrected_y_vel = y_vel  # Keep as processed (already negated in script)
        else:  # Bottom lanes (going right)
            # Y velocity normal
            # In HighD bottom lanes: positive Y = moving toward top of screen
            # For lane changes: positive Y should mean moving to left lane
            corrected_y_vel = y_vel

        return x_vel, corrected_y_vel

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

    def create_historical_positions(self, input_frames: List[Tuple], traffic_direction: int) -> str:
        """Create 5 historical positions from [-4s, -2s] period at 0.4s intervals"""
        positions = []

        # Input frames cover [-4s, -2s] = 50 frames (2 seconds at 25 Hz)
        # We want 5 points at 0.4s intervals = every 10 frames
        if len(input_frames) < 50:
            padded_frames = input_frames + [input_frames[-1]] * (50 - len(input_frames))
        else:
            padded_frames = input_frames[:50]

        # Use the LAST frame as reference point (current position at -2s before crossing)
        reference_frame = padded_frames[-1]
        ref_x = reference_frame[self.X_POSITION]  # HighD coordinates (meters)
        ref_y = reference_frame[self.Y_POSITION]  # HighD coordinates (meters)

        for i in range(5):
            frame_idx = i * 10  # 0, 10, 20, 30, 40 (every 0.4s)
            frame = padded_frames[frame_idx]

            # Get raw positions in HighD coordinates
            raw_x = frame[self.X_POSITION]
            raw_y = frame[self.Y_POSITION]

            # Calculate relative movement in vehicle-centric coordinates
            # For LC-LLM: X = forward direction, Y = lateral direction

            if traffic_direction == 1:  # Top lanes (going left in HighD)
                # In HighD top lanes: negative X = forward movement
                # Transform to LC-LLM coordinates where positive X = forward
                rel_x = ref_x - raw_x  # Forward movement (negative HighD X becomes positive)
                rel_y = raw_y - ref_y  # Lateral movement (positive = left)
            else:  # Bottom lanes (going right in HighD)
                # In HighD bottom lanes: positive X = forward movement
                rel_x = raw_x - ref_x  # Forward movement
                rel_y = raw_y - ref_y  # Lateral movement (positive = left)

            positions.append(f"({rel_x:.2f},{rel_y:.2f})")

        return ", ".join(positions)

    def create_surrounding_vehicles_info(self, current_frame: Tuple) -> List[str]:
        """Create surrounding vehicle info using ONLY distance data"""
        surrounding_info = []

        distance_threshold = 200  # meters

        # Ahead vehicle
        if current_frame[self.PRECEDING_DISTANCE] < distance_threshold:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            distance = current_frame[self.PRECEDING_DISTANCE]
            surrounding_info.append(f"- Ahead: a {vehicle_type} at {distance:.0f} m distance.")

        # Behind vehicle
        if current_frame[self.FOLLOWING_DISTANCE] < distance_threshold:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            distance = current_frame[self.FOLLOWING_DISTANCE]
            surrounding_info.append(f"- Behind: a {vehicle_type} at {distance:.0f} m distance.")

        # Left front
        if current_frame[self.LEFT_PRECEDING_DISTANCE] < distance_threshold:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            distance = current_frame[self.LEFT_PRECEDING_DISTANCE]
            surrounding_info.append(f"- Left front: a {vehicle_type} at {distance:.0f} m distance.")

        # Right front
        if current_frame[self.RIGHT_PRECEDING_DISTANCE] < distance_threshold:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            distance = current_frame[self.RIGHT_PRECEDING_DISTANCE]
            surrounding_info.append(f"- Right front: a {vehicle_type} at {distance:.0f} m distance.")

        # Left rear
        if current_frame[self.LEFT_FOLLOWING_DISTANCE] < distance_threshold:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            distance = current_frame[self.LEFT_FOLLOWING_DISTANCE]
            surrounding_info.append(f"- Left rear: a {vehicle_type} at {distance:.0f} m distance.")

        # Right rear
        if current_frame[self.RIGHT_FOLLOWING_DISTANCE] < distance_threshold:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            distance = current_frame[self.RIGHT_FOLLOWING_DISTANCE]
            surrounding_info.append(f"- Right rear: a {vehicle_type} at {distance:.0f} m distance.")

        return surrounding_info

    def generate_notable_features(self, current_frame: Tuple, lane_position: str, traffic_direction: int) -> List[str]:
        """Generate notable features based on vehicle state"""
        features = []

        # Get corrected velocities
        vx_ms, vy_ms = self.get_corrected_velocities(current_frame, traffic_direction)
        vx_kmh = abs(vx_ms) * 3.6  # Use absolute value for speed
        vy_kmh = vy_ms * 3.6

        # Lateral movement (reliable indicator)
        if abs(vy_kmh) > 2.0:  # Significant lateral movement
            direction = "left" if vy_kmh > 0 else "right"
            features.append(
                f"Notable feature: Significant lateral movement detected ({direction}ward at {abs(vy_kmh):.1f} km/h).")

        # Longitudinal acceleration
        ax = current_frame[self.X_ACCELERATION]
        if abs(ax) > 0.5:
            accel_type = "acceleration" if ax > 0 else "deceleration"
            features.append(f"Notable feature: Notable {accel_type} (ax = {ax:.2f} m/s²).")

        # Distance-based analysis
        ahead_distance = current_frame[self.PRECEDING_DISTANCE]
        if ahead_distance < 50:
            features.append("Notable feature: Vehicle ahead is very close, ahead is blocked.")
        elif ahead_distance < 100:
            features.append("Notable feature: Vehicle ahead is moderately close, ahead is partially blocked.")
        else:
            features.append("Notable feature: Ahead is clear.")

        # Lane availability analysis
        if current_frame[self.LEFT_LANE_EXIST] > 0.5:
            left_front_dist = current_frame[self.LEFT_PRECEDING_DISTANCE]
            if left_front_dist > 100:
                features.append("Notable feature: Left lane is clear.")
            else:
                features.append(f"Notable feature: Left lane has vehicle at {left_front_dist:.0f}m.")

        if current_frame[self.RIGHT_LANE_EXIST] > 0.5:
            right_front_dist = current_frame[self.RIGHT_PRECEDING_DISTANCE]
            if right_front_dist > 100:
                features.append("Notable feature: Right lane is clear.")
            else:
                features.append(f"Notable feature: Right lane has vehicle at {right_front_dist:.0f}m.")

        # Vehicle type
        if current_frame[self.CAR_TYPE] < 0:  # Truck
            features.append("Notable feature: The target vehicle is a Truck.")

        # Truck ahead
        if ahead_distance < 100 and current_frame[self.CAR_TYPE] < 0:
            features.append("Notable feature: Large vehicle (Truck) ahead.")

        # Lane position
        features.append(f"Notable feature: Vehicle in {lane_position} lane.")

        return features

    def determine_potential_behavior(self, current_frame: Tuple, intention: int, lane_position: str) -> str:
        """Determine potential behavior based on intention and spatial context"""
        ahead_distance = current_frame[self.PRECEDING_DISTANCE]

        if intention == 0:  # Keep lane
            if ahead_distance < 60:
                return "Following and keep lane"
            else:
                return "Normal keep lane"

        elif intention == 1:  # Left lane change
            if ahead_distance < 80 and lane_position in ["rightmost", "middle"]:
                return "Change to the left lane for overtaking"
            else:
                return "Left lane change maneuver"

        else:  # intention == 2, Right lane change
            if ahead_distance < 80 and lane_position in ["leftmost", "middle"]:
                return "Change to the right lane for overtaking"
            elif current_frame[self.CAR_TYPE] < 0:  # Truck
                return "Change to the right lane (truck behavior)"
            else:
                return "Right lane change maneuver"

    def extract_ground_truth_trajectory(self, future_frames: List[Tuple], reference_frame: Tuple,
                                        traffic_direction: int) -> str:
        """Extract actual future trajectory from ground truth data [0s, 4s] at 0.2s intervals (20 points)"""
        trajectory_points = []

        # Reference position at crossing time (should be (0,0) in relative coordinates)
        ref_x = reference_frame[self.X_POSITION]
        ref_y = reference_frame[self.Y_POSITION]

        # Future frames cover [0s, 4s] = 100 frames at 25 Hz
        # Sample at 0.2s intervals = every 5 frames: 0.2s, 0.4s, 0.6s, ..., 4.0s
        # Frame indices for 20 points: 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99
        sample_indices = [4 + i * 5 for i in range(20)]  # Generate 20 indices at 5-frame intervals

        for i, frame_idx in enumerate(sample_indices):
            if frame_idx < len(future_frames):
                future_frame = future_frames[frame_idx]

                # Get raw future position
                raw_x = future_frame[self.X_POSITION]
                raw_y = future_frame[self.Y_POSITION]

                # Transform to vehicle-centric coordinates (same logic as historical)
                if traffic_direction == 1:  # Top lanes (going left in HighD)
                    rel_x = ref_x - raw_x  # Forward movement
                    rel_y = raw_y - ref_y  # Lateral movement
                else:  # Bottom lanes (going right in HighD)
                    rel_x = raw_x - ref_x  # Forward movement
                    rel_y = raw_y - ref_y  # Lateral movement

                trajectory_points.append(f"({rel_x:.2f},{rel_y:.2f})")
            else:
                # Fallback
                if trajectory_points:
                    trajectory_points.append(trajectory_points[-1])
                else:
                    trajectory_points.append("(0.00,0.00)")

        # Ensure exactly 20 points
        while len(trajectory_points) < 20:
            if trajectory_points:
                trajectory_points.append(trajectory_points[-1])
            else:
                trajectory_points.append("(0.00,0.00)")

        return "[" + ", ".join(trajectory_points[:20]) + "]"

    def convert_sample_to_lcllm_format(self, features_sequence: List[Tuple], direction_labels: List[int]) -> Dict:
        """Convert trajectory sample to LC-LLM format with HighD awareness"""

        if len(features_sequence) != 200:
            raise ValueError(f"Expected 200 frames, got {len(features_sequence)}")

        # Determine traffic direction from velocity pattern
        traffic_direction = self.determine_traffic_direction(features_sequence)

        crossing_idx = 100  # Frame 100 is crossing time

        # Input period: [-4s, -2s] = frames 0-49 (first 50 frames)
        input_frames = features_sequence[:50]

        # Current frame: last frame of input period (at -2s)
        current_frame = input_frames[-1]  # Frame 49

        # Reference frame: at crossing time (frame 100)
        reference_frame = features_sequence[crossing_idx]

        # Future period: [0s, 4s] = frames 100-199
        future_frames = features_sequence[crossing_idx:]

        # Determine intention from future period labels
        future_labels = direction_labels[crossing_idx:]
        if future_labels:
            intention = max(set(future_labels), key=future_labels.count)
        else:
            intention = 0

        # Generate scenario components
        lane_position, lane_count = self.determine_lane_configuration(current_frame)

        # Vehicle information (get corrected velocities)
        car_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
        vx_ms, vy_ms = self.get_corrected_velocities(current_frame, traffic_direction)
        vx_kmh = abs(vx_ms) * 3.6  # Speed (always positive)
        vy_kmh = vy_ms * 3.6  # Lateral velocity (can be negative)
        ax = current_frame[self.X_ACCELERATION]
        ay = current_frame[self.Y_ACCELERATION]

        # Vehicle dimensions (reasonable estimates)
        if car_type == "Car":
            width, length = np.random.uniform(1.8, 2.2), np.random.uniform(4.2, 5.5)
        else:
            width, length = 2.5, np.random.uniform(12.0, 22.0)

        # Generate components
        historical_positions = self.create_historical_positions(input_frames, traffic_direction)
        surrounding_vehicles = self.create_surrounding_vehicles_info(current_frame)
        notable_features = self.generate_notable_features(current_frame, lane_position, traffic_direction)
        potential_behavior = self.determine_potential_behavior(current_frame, intention, lane_position)
        future_trajectory = self.extract_ground_truth_trajectory(future_frames, reference_frame, traffic_direction)

        # Create scenario description
        lane_description = f"a {lane_count}-lane highway" if lane_count > 1 else "highway"
        scenario = f"""The target vehicle is driving on {lane_description}, located at the {lane_position} lane.
The information of target vehicle is as follow:
- Velocity(km/h): vx={vx_kmh:.2f}, vy={vy_kmh:.2f}
- Acceleration(m/s^2): ax={ax:.2f}, ay={ay:.2f}
- Type: {car_type}, with width of {width:.2f} m and length of {length:.2f} m
- Historical position of the last 2 seconds (One point every 0.4s): [{historical_positions}]
The information of its surrounding vehicles (within detection range) are listed as follow:
{chr(10).join(surrounding_vehicles) if surrounding_vehicles else "- No surrounding vehicles detected within range."}"""

        # Map intention
        intention_map = {0: "0: Keep lane", 1: "1: Left lane change", 2: "2: Right lane change"}
        intention_text = intention_map[intention]

        # Create reasoning and response
        thought_section = "Thought:\n- " + "\n- ".join(notable_features)
        thought_section += f"\n- Potential behavior: {potential_behavior}."

        response = f"""{thought_section}
Final Answer:
- Intention: "{intention_text}"
- Trajectory: "{future_trajectory}"
 """

        # Create final Llama format
        llama_sample = {
            "text": f"<s>[INST] <<SYS>>\n{self.SYSTEM_MESSAGE}\n<</SYS>>\n\n{scenario} [/INST] {response}</s>"
        }

        return llama_sample


def main():
    """Convert your HighD pickle data to LC-LLM format"""
    import pickle
    import glob
    import os

    converter = FixedLCLLMDataConverter()

    # Process your data from output_4sbefore_4safter
    data_dir = "output_4sbefore_4safter"
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        return

    pickle_files = glob.glob(f"{data_dir}/*.pickle")
    pickle_files.sort()

    print(f"Found {len(pickle_files)} pickle files in {data_dir}")
    print("Processing files: first 50 for training, last 10 for testing...")

    # Split into training (first 50) and testing (last 10)
    train_files = pickle_files[:50]
    test_files = pickle_files[50:60] if len(pickle_files) >= 60 else pickle_files[50:]

    print(f"Training files: {len(train_files)}")
    print(f"Testing files: {len(test_files)}")

    # Process training data
    train_samples = []
    total_train_processed = 0

    for i, file_path in enumerate(train_files, 1):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        sample_count = 0
        for features_sequence, direction_labels in data:
            if len(features_sequence) == 200:
                try:
                    sample = converter.convert_sample_to_lcllm_format(
                        features_sequence, direction_labels
                    )
                    train_samples.append(sample)
                    sample_count += 1

                except Exception as e:
                    print(f"Error in training file {i}: {e}")
                    continue

        total_train_processed += sample_count
        if i % 10 == 0 or i == len(train_files):  # Progress every 10 files
            print(f"Training: {i}/{len(train_files)} files ({total_train_processed} samples)")

    # Process testing data
    test_samples = []
    total_test_processed = 0

    for i, file_path in enumerate(test_files, 1):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        sample_count = 0
        for features_sequence, direction_labels in data:
            if len(features_sequence) == 200:
                try:
                    sample = converter.convert_sample_to_lcllm_format(
                        features_sequence, direction_labels
                    )
                    test_samples.append(sample)
                    sample_count += 1

                except Exception as e:
                    print(f"Error in testing file {i}: {e}")
                    continue

        total_test_processed += sample_count

    print(f"Testing: {len(test_files)}/{len(test_files)} files ({total_test_processed} samples)")

    # Save training data
    train_output_file = "lcllm_training_data_20points.json"
    with open(train_output_file, 'w') as f:
        json.dump(train_samples, f, indent=2)

    # Save testing data
    test_output_file = "lcllm_testing_data_20points.json"
    with open(test_output_file, 'w') as f:
        json.dump(test_samples, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"✓ Training: {len(train_samples)} samples → {train_output_file}")
    print(f"✓ Testing: {len(test_samples)} samples → {test_output_file}")
    print(f"✓ Total: {len(train_samples) + len(test_samples)} samples from {len(pickle_files)} files")

    # Show brief example from training data
    if train_samples:
        print(f"\nTraining sample preview:")
        example_text = train_samples[0]["text"]
        if "[/INST]" in example_text:
            input_part = example_text.split("[/INST]")[0].replace("<s>[INST]", "").strip()
            print(f"Length: {len(example_text)} characters")
            print(f"Input preview: {input_part[:200]}...")
        else:
            print(f"Length: {len(example_text)} characters")
            print(f"Preview: {example_text[:200]}...")


if __name__ == "__main__":
    main()