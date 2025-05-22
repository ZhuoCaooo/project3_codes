#!/usr/bin/env python3
"""
Convert numerical trajectory data to LC-LLM format (matching actual paper format)
Based on analysis of real LC-LLM training data
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
        """FIXED: Create 6 historical positions properly"""
        positions = []

        if len(history_frames) < 6:
            # Pad with the first frame if not enough history
            padded_frames = [history_frames[0]] * (6 - len(history_frames)) + history_frames
        else:
            # Take last 6 frames
            padded_frames = history_frames[-6:]

        # Generate positions going backwards in time
        for i, frame in enumerate(padded_frames):
            time_offset = (5 - i) * 0.4  # 0.4s intervals, going backwards
            x_velocity = frame[self.X_VELOCITY]  # KEEP sign for direction
            x_pos = -time_offset * x_velocity  # Negative offset (going back in time)
            y_pos = frame[self.DELTA_Y]

            positions.append(f"({x_pos:.2f},{y_pos:.2f})")

        return ", ".join(positions)

    def create_surrounding_vehicles_info(self, current_frame: Tuple) -> List[str]:
        """FIXED: Better surrounding vehicle info"""
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
            speed = ego_velocity_kmh - (10 * (1 if ego_velocity_kmh > 0 else -1))  # ← FIXED
            surrounding_info.append(
                f"- Behind: a {vehicle_type} traveling at {speed:.2f} km/h of X-axis, "
                f"with a distance of {current_frame[self.FOLLOWING_TTC]:.0f} m."
            )

        # Left front
        if current_frame[self.LEFT_PRECEDING_TTC] < 200:
            vehicle_type = "Car" if current_frame[self.CAR_TYPE] > 0 else "Truck"
            speed = ego_velocity_kmh + (15 * (1 if ego_velocity_kmh > 0 else -1))  # ← Make sure this is correct too
            surrounding_info.append(
                f"- Left front: a {vehicle_type} traveling at {speed:.2f} km/h of X-axis, "
                f"with a distance of {current_frame[self.LEFT_PRECEDING_TTC]:.0f} m."
            )

        # Add more vehicles for realism...

        return surrounding_info

    def generate_notable_features(self, current_frame: Tuple, lane_position: str) -> List[str]:
        """FIXED: Better logic for vehicle analysis"""
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

        # FIXED: Ahead vehicle analysis
        if current_frame[self.PRECEDING_TTC] < 200:
            if current_frame[self.PRECEDING_TTC] < 80:  # Close vehicle
                features.append("Notable feature: Ahead is block.")
            else:
                features.append("Notable feature: Ahead is free.")
        else:
            features.append("Notable feature: Ahead is free.")

        # FIXED: Left lane analysis with proper speed comparison
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

    def calculate_future_trajectory(self, current_frame: Tuple,
                                    future_frames: List[Tuple], intention: int) -> str:
        """FIXED: Proper trajectory calculation - preserve direction"""
        trajectory_points = []

        # KEEP negative velocities - they indicate direction!
        vx = current_frame[self.X_VELOCITY]  # Preserve sign for direction
        vy = current_frame[self.Y_VELOCITY]
        current_y = current_frame[self.DELTA_Y]

        for i in range(4):  # 4 points, 1 second each
            time_step = i + 1

            # X position - preserve direction (negative = leftward, positive = rightward)
            x_pos = vx * time_step

            # Y position based on intention
            if i < len(future_frames):
                # Use actual future position if available
                y_pos = future_frames[i][self.DELTA_Y]
            else:
                # Estimate based on intention
                if intention == 1:  # Left change
                    y_pos = current_y + (time_step * 0.8)  # Gradual left movement
                elif intention == 2:  # Right change
                    y_pos = current_y - (time_step * 0.8)  # Gradual right movement
                else:  # Keep lane
                    y_pos = current_y + vy * time_step

            trajectory_points.append(f"({x_pos:.2f},{y_pos:.2f})")

        return "[" + ", ".join(trajectory_points) + "]"

    def convert_sample_to_lcllm_format(self, features_sequence: List[Tuple],
                                       direction_labels: List[int]) -> Dict:
        """Convert trajectory sample to exact LC-LLM format"""

        # Use middle frame as current
        current_idx = len(features_sequence) // 2
        current_frame = features_sequence[current_idx]

        # Get history and future
        history_frames = features_sequence[max(0, current_idx - 20):current_idx]
        future_frames = features_sequence[current_idx + 1:current_idx + 5]

        # Determine intention (most common in future)
        future_labels = direction_labels[current_idx:]
        intention = max(set(future_labels), key=future_labels.count) if future_labels else 0

        # Generate scenario components
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

        # Historical positions
        historical_positions = self.create_historical_positions(history_frames)

        # Surrounding vehicles
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

        # Generate reasoning
        notable_features = self.generate_notable_features(current_frame, lane_position)
        potential_behavior = self.determine_potential_behavior(current_frame, intention, lane_position)
        future_trajectory = self.calculate_future_trajectory(current_frame, future_frames, intention)

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
    """Convert your pickle data to LC-LLM format"""
    import pickle
    import glob

    converter = LCLLMDataConverter()

    # Process your data
    pickle_files = glob.glob("output_8sbefore_2safter/*.pickle")
    all_samples = []

    for file_path in pickle_files[:45]:  # Process first 3 files
        print(f"Processing {file_path}...")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for features_sequence, direction_labels in data:
            if len(features_sequence) >= 50:  # Minimum length check
                try:
                    sample = converter.convert_sample_to_lcllm_format(
                        features_sequence, direction_labels
                    )
                    all_samples.append(sample)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue

        print(f"Converted {len(data)} samples from {file_path}")

    # Save result
    with open("lcllm_training_data.json", 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"Saved {len(all_samples)} samples to lcllm_training_data.json")

    # Show example
    if all_samples:
        print("\n" + "=" * 70)
        print("EXAMPLE OUTPUT (matches paper format):")
        print("=" * 70)
        example_text = all_samples[0]["text"]
        # Pretty print by splitting at key sections
        parts = example_text.split("[/INST]")
        print("INPUT:", parts[0].replace("<s>[INST]", "").strip()[:500] + "...")
        print("\nOUTPUT:", parts[1].replace("</s>", "").strip()[:300] + "...")


if __name__ == "__main__":
    main()