from config import SYSTEM_MESSAGE
import numpy as np

# Feature names based on the dataset description
FEATURE_NAMES = [
    "Left_Lane_Exists", "Right_Lane_Exists", "Delta_Y",
    "Y_Velocity", "Y_Acceleration", "X_Velocity",
    "X_Acceleration", "Vehicle_Type",
    "Preceding_Vehicle_Distance", "Following_Vehicle_Distance",
    "Left_Preceding_Distance", "Left_Alongside_Distance", "Left_Following_Distance",
    "Right_Preceding_Distance", "Right_Alongside_Distance", "Right_Following_Distance"
]


def get_lane_description(features):
    """Generate lane description from features."""
    left_lane_exists = features[0] > 0.5
    right_lane_exists = features[1] > 0.5

    if left_lane_exists and right_lane_exists:
        return "middle"
    elif left_lane_exists:
        return "rightmost"
    elif right_lane_exists:
        return "leftmost"
    else:
        return "single"  # Unlikely case


def get_vehicle_type(features):
    """Get vehicle type based on Vehicle_Type feature."""
    vehicle_type = features[7]
    return "Car" if vehicle_type > 0 else "Truck"


def format_historical_positions(history_frames):
    """Format historical positions for the prompt."""
    positions = []
    for frame in history_frames:
        # We use Delta_Y as the lateral position
        positions.append(f"({-frame[2]:.2f},{frame[3]:.2f})")

    return ", ".join(positions)


def create_surrounding_vehicle_info(features):
    """Create description of surrounding vehicles."""
    info = []

    # Process preceding vehicle (ahead)
    preceding_dist = features[8]
    if preceding_dist < 200:  # If vehicle exists within range
        info.append(f"- Ahead: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {preceding_dist:.0f} m.")

    # Process following vehicle (behind)
    following_dist = features[9]
    if following_dist < 200:
        info.append(f"- Behind: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {following_dist:.0f} m.")

    # Process left preceding (left front)
    left_preceding_dist = features[10]
    if left_preceding_dist < 200:
        info.append(f"- Left front: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {left_preceding_dist:.0f} m.")

    # Process left alongside
    left_alongside_dist = features[11]
    if left_alongside_dist < 200:
        info.append(f"- Left side: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {left_alongside_dist:.0f} m.")

    # Process left following (left rear)
    left_following_dist = features[12]
    if left_following_dist < 200:
        info.append(f"- Left rear: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {left_following_dist:.0f} m.")

    # Process right preceding (right front)
    right_preceding_dist = features[13]
    if right_preceding_dist < 200:
        info.append(f"- Right front: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {right_preceding_dist:.0f} m.")

    # Process right alongside
    right_alongside_dist = features[14]
    if right_alongside_dist < 200:
        info.append(f"- Right side: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {right_alongside_dist:.0f} m.")

    # Process right following (right rear)
    right_following_dist = features[15]
    if right_following_dist < 200:
        info.append(f"- Right rear: a {get_vehicle_type(features)} traveling at {features[5]:.2f} km/h of X-axis, "
                    f"with a distance of {right_following_dist:.0f} m.")

    return "\n".join(info)


def create_prompt(current_frame, history_frames, intention, future_frames):
    """Create a prompt for the LC-LLM model based on the features."""

    # Generate lane description
    lane_position = get_lane_description(current_frame)
    lane_count = 1
    if current_frame[0] > 0.5:  # Left lane exists
        lane_count += 1
    if current_frame[1] > 0.5:  # Right lane exists
        lane_count += 1

    # Create user message
    user_message = f"""The target vehicle is driving on a {lane_count}-lane highway, located at the {lane_position} lane.
The information of target vehicle is as follow:
- Velocity(km/h): vx={current_frame[5]:.2f}, vy={current_frame[3]:.2f}
- Accelaration(m/s^2): ax={current_frame[6]:.2f}, ay={current_frame[4]:.2f}
- Type: {get_vehicle_type(current_frame)}, with width of 1.92 m and length of 4.24 m
- Historical position of the last 2 seconds (One point every 0.4s): [{format_historical_positions(history_frames)}]
The information of its surrounding vehicles (with a range of 200m) are listed as follow:
{create_surrounding_vehicle_info(current_frame)}
"""

    # For training, we need to include the ground truth output
    # Format the future trajectory
    future_positions = []
    for i, frame in enumerate(future_frames):
        # Use delta_y and y_velocity as the trajectory coordinates
        # In the actual implementation you might want to use real positions
        future_positions.append(f"({(i + 1) * 25:.2f},{frame[2]:.2f})")

    trajectory_str = ", ".join(future_positions)

    # Map intention to text
    intention_map = {0: "0: Keep lane", 1: "1: Left lane change", 2: "2: Right lane change"}
    intention_text = intention_map[intention]

    # Include Chain-of-Thought reasoning (this is simplified and would be more elaborate in real implementation)
    cot_reasoning = ""
    if intention == 0:
        cot_reasoning = """Thought:
- Notable feature: Ahead is free.
- Notable feature: The speed of surrounding vehicles is similar to the target vehicle's.
- Potential behavior: Normal Keep lane."""
    elif intention == 1:
        cot_reasoning = """Thought:
- Notable feature: The speed of the ahead vehicle is slower than that of the target's.
- Notable feature: Left front is free.
- Potential behavior: Change to the left lane for overtaking."""
    else:  # intention == 2
        cot_reasoning = """Thought:
- Notable feature: The speed of the ahead vehicle is slower than that of the target's.
- Notable feature: Right front is free.
- Potential behavior: Change to the right lane for overtaking."""

    # Compose the complete model output
    model_output = f"""{cot_reasoning}
Final Answer:
- Intention: "{intention_text}"
- Trajectory: "[{trajectory_str}]"
"""

    # Combine system message, user message, and model output
    full_prompt = f"{SYSTEM_MESSAGE}\n{user_message}\n{model_output}"

    return full_prompt