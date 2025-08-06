def transform_to_vehicle_centric_coordinates(tracks_csv, tracks_meta, original_lane, vehicle_id, frame_num):
    """
    Transform HighD global coordinates to LC-LLM vehicle-centric coordinates
    """

    # Get raw HighD data
    raw_x = tracks_csv[vehicle_id]['x'][frame_num]
    raw_y = tracks_csv[vehicle_id]['y'][frame_num]
    raw_x_vel = tracks_csv[vehicle_id]['xVelocity'][frame_num]
    raw_y_vel = tracks_csv[vehicle_id]['yVelocity'][frame_num]
    raw_x_acc = tracks_csv[vehicle_id]['xAcceleration'][frame_num]
    raw_y_acc = tracks_csv[vehicle_id]['yAcceleration'][frame_num]

    # Determine vehicle direction
    is_upper_lane = original_lane in [2, 3, 4, 5]  # Moving LEFT
    is_lower_lane = original_lane in [6, 7, 8, 9]  # Moving RIGHT

    if is_upper_lane:
        # UPPER LANES: Vehicles moving LEFT (X decreases in HighD)
        # Transform to vehicle-centric: X should increase forward

        # Method 1: Flip X direction
        transformed_x = -raw_x  # Now X increases as vehicle moves forward
        transformed_x_vel = -raw_x_vel  # Positive = forward motion
        transformed_x_acc = -raw_x_acc  # Positive = acceleration

        # Y direction: In HighD, +Y = downward. For upper lanes moving left:
        # Vehicle's left side = upward in image = negative Y in HighD
        # So for vehicle-centric: positive Y = left side = negative HighD Y
        transformed_y = -raw_y  # Flip Y: positive = left side of vehicle
        transformed_y_vel = -raw_y_vel  # Positive = moving to vehicle's left
        transformed_y_acc = -raw_y_acc  # Positive = accelerating left

    else:  # LOWER LANES
        # LOWER LANES: Vehicles moving RIGHT (X increases in HighD)
        # Already correct direction for longitudinal

        transformed_x = raw_x  # Keep as is - already increases forward
        transformed_x_vel = raw_x_vel  # Positive = forward motion
        transformed_x_acc = raw_x_acc  # Positive = acceleration

        # Y direction: For lower lanes moving right:
        # Vehicle's left side = upward in image = negative Y in HighD
        # So same transformation as upper lanes
        transformed_y = -raw_y  # Flip Y: positive = left side of vehicle
        transformed_y_vel = -raw_y_vel  # Positive = moving to vehicle's left
        transformed_y_acc = -raw_y_acc  # Positive = accelerating left

    return {
        'x': transformed_x,
        'y': transformed_y,
        'x_velocity': transformed_x_vel,
        'y_velocity': transformed_y_vel,
        'x_acceleration': transformed_x_acc,
        'y_acceleration': transformed_y_acc,
        'lane_type': 'UPPER' if is_upper_lane else 'LOWER'
    }


# Test the transformation
def test_coordinate_transformation(track_number="01"):
    """Test coordinate transformation on sample vehicles"""

    tracks_csv = read_tracks_csv(f"data/{track_number}_tracks.csv")
    tracks_meta = read_tracks_meta(f"data/{track_number}_tracksMeta.csv")

    print("=== Coordinate Transformation Test ===")

    # Test vehicles from different lanes
    test_vehicles = {5: 1, 2: 2, 3: 4, 6: 6}  # lane: vehicle_id

    for lane, vid in test_vehicles.items():
        print(f"\n--- Lane {lane} Vehicle {vid} ---")

        # Test first 5 frames
        for frame in range(5):
            original = {
                'x': tracks_csv[vid]['x'][frame],
                'x_vel': tracks_csv[vid]['xVelocity'][frame]
            }

            transformed = transform_to_vehicle_centric_coordinates(
                tracks_csv, tracks_meta, lane, vid, frame
            )

            print(f"Frame {frame}: X {original['x']:6.1f}→{transformed['x']:6.1f}, "
                  f"X_vel {original['x_vel']:+5.2f}→{transformed['x_velocity']:+5.2f}")


if __name__ == "__main__":
    test_coordinate_transformation("01")