from read_data import *


def test_consistent_fix(track_number="01"):
    """Test the fixed consistent calculation"""

    tracks_csv = read_tracks_csv(f"data/{track_number}_tracks.csv")
    tracks_meta = read_tracks_meta(f"data/{track_number}_tracksMeta.csv")
    recording_meta = read_recording_meta(f"data/{track_number}_recordingMeta.csv")

    # Your existing setup
    lanes_info = {}
    lanes_info[2] = recording_meta['upperLaneMarkings'][0]
    lanes_info[3] = recording_meta['upperLaneMarkings'][1]
    lanes_info[5] = recording_meta['lowerLaneMarkings'][0]
    lanes_info[6] = recording_meta['lowerLaneMarkings'][1]
    lane_width = 4.02

    print("=== FIXED vs ORIGINAL Comparison ===")

    # Test vehicles
    test_vehicles = {5: 1, 2: 2, 3: 4, 6: 6}  # lane: vehicle_id

    for lane, vid in test_vehicles.items():
        car_center = tracks_csv[vid]['y'][0] + tracks_meta[vid]['height'] / 2
        raw_y_vel = tracks_csv[vid]['yVelocity'][5]

        # ORIGINAL (your current buggy method)
        if lane in [2, 3]:  # Upper
            original_delta_y = car_center - lanes_info[lane] - lane_width / 2
            original_y_vel = -raw_y_vel
        else:  # Lower
            original_delta_y = lanes_info[lane] - car_center + lane_width / 2
            original_y_vel = raw_y_vel

        # FIXED (consistent method)
        fixed_delta_y = car_center - (lanes_info[lane] + lane_width / 2)
        fixed_y_vel = raw_y_vel  # Keep raw velocity

        direction = "UP/LEFT" if lane in [2, 3] else "DOWN/RIGHT"
        print(f"Lane {lane} ({direction}):")
        print(f"  Delta Y: Original={original_delta_y:+.2f}, Fixed={fixed_delta_y:+.2f}")
        print(f"  Y Vel:   Original={original_y_vel:+.2f}, Fixed={fixed_y_vel:+.2f}")
        print()


if __name__ == "__main__":
    test_consistent_fix("01")