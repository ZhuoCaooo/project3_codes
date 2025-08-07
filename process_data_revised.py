from read_data import *
import random



'''this script involves a complex coordination transformation process, but it has problems!!!!!!!!!!!!!!!!! dont use'''
# HighD frame_rate = 25 Hz
TRAJECTORY_LENGTH = 200  # Total trajectory length: 6 seconds (150 frames at 25 Hz)
FRAMES_AFTER_CROSSING = 100  # Include 50 frames (2 seconds) after crossing
INCLUDE_AFTER_CROSSING = True  # Use the FRAMES_AFTER_CROSSING parameter


def run(number):
    '''
    This function runs the data processing code and output to pickle files
    '''
    # read from 3 files
    tracks_csv = read_tracks_csv("data/" + number + "_tracks.csv")
    tracks_meta = read_tracks_meta("data/" + number + "_tracksMeta.csv")
    recording_meta = read_recording_meta(
        "data/" + number + "_recordingMeta.csv")

    # figure out the lane changing cars and lane keeping cars
    lane_changing_ids = []
    lane_keeping_ids = []
    for key in tracks_meta:
        if (tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
            lane_changing_ids.append(key)
        else:
            lane_keeping_ids.append(key)

    # get the lane information
    lanes_info = {}
    lane_num = len(recording_meta[UPPER_LANE_MARKINGS]) + \
               len(recording_meta[LOWER_LANE_MARKINGS]) - 2
    if lane_num == 4:
        # 4 lanes
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[5] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][1]
        lane_width = ((lanes_info[3] - lanes_info[2]) +
                      (lanes_info[6] - lanes_info[5])) / 2
    elif lane_num == 6:
        # 6 lanes
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
        lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][1]
        lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][2]
        lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) +
                      (lanes_info[7] - lanes_info[6]) + (lanes_info[8] - lanes_info[7])) / 4
    elif lane_num == 7:
        # 7 lanes: track 58 ~ 60
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
        lanes_info[5] = recording_meta[UPPER_LANE_MARKINGS][3]
        lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][1]
        lanes_info[9] = recording_meta[LOWER_LANE_MARKINGS][2]
        lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) + (
                lanes_info[5] - lanes_info[4]) + (lanes_info[8] - lanes_info[7]) + (lanes_info[9] - lanes_info[8])) / 5
    else:
        print("Error: Invalid input -", number)

    def determine_lane_exist(cur_lane):
        '''
        return: left_exist, right_exist
        Have to do this in a hardcoded way to determine the existence of neighbor lanes.
        '''
        if lane_num == 4:
            if cur_lane == 2 or cur_lane == 6:
                return 1, 0
            else:
                return 0, 1
        elif lane_num == 6:
            if cur_lane == 2 or cur_lane == 8:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 7:
                return 1, 1
            else:
                return 0, 1
        elif lane_num == 7:
            if cur_lane == 2 or cur_lane == 9:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 4 or cur_lane == 8:
                return 1, 1
            else:
                return 0, 1

    def transform_coordinates_to_vehicle_centric(i, frame_num, original_lane):
        '''
        CORRECTED VERSION: Consistent Y transformation for all vehicles

        Based on test results, the key insight is:
        - Both upper and lower lane vehicles need Y flipped the SAME way
        - Y increases in HighD = right-relative-to-driving for ALL vehicles
        - We want positive Y = left-relative-to-driving for ALL vehicles
        - So we flip Y for ALL vehicles consistently
        '''
        # Get raw HighD data
        raw_x = tracks_csv[i][X][frame_num]
        raw_y = tracks_csv[i][Y][frame_num]
        raw_x_vel = tracks_csv[i][X_VELOCITY][frame_num]
        raw_y_vel = tracks_csv[i][Y_VELOCITY][frame_num]
        raw_x_acc = tracks_csv[i][X_ACCELERATION][frame_num]
        raw_y_acc = tracks_csv[i][Y_ACCELERATION][frame_num]

        # Determine driving direction from tracks_meta
        driving_direction = tracks_meta[i][DRIVING_DIRECTION]

        if driving_direction == 1:  # Upper lanes (vehicles moving LEFT in HighD)
            # Flip X so they move in positive direction
            transformed_x = -raw_x
            transformed_x_vel = -raw_x_vel
            transformed_x_acc = -raw_x_acc

            # Flip Y so positive Y = left-relative-to-driving
            transformed_y = -raw_y
            transformed_y_vel = -raw_y_vel
            transformed_y_acc = -raw_y_acc

        else:  # Lower lanes (vehicles moving RIGHT in HighD)
            # Keep X as is (already moving in positive direction)
            transformed_x = raw_x
            transformed_x_vel = raw_x_vel
            transformed_x_acc = raw_x_acc

            # Flip Y so positive Y = left-relative-to-driving (SAME flip as upper!)
            transformed_y = -raw_y
            transformed_y_vel = -raw_y_vel
            transformed_y_acc = -raw_y_acc

        return {
            'x': transformed_x,
            'y': transformed_y,
            'x_velocity': transformed_x_vel,
            'y_velocity': transformed_y_vel,
            'x_acceleration': transformed_x_acc,
            'y_acceleration': transformed_y_acc
        }

    # Also update your lane center calculation to be consistent:
    def get_corrected_lane_center(original_lane, lanes_info, lane_width):
        """
        Get lane center in vehicle-centric coordinates (consistent with Y flip)
        """
        # Lane center in HighD coordinates
        lane_center_highd = lanes_info[original_lane] + lane_width / 2

        # Apply SAME Y transformation as coordinates: flip Y
        lane_center_vehicle = -lane_center_highd

        return lane_center_vehicle

    def construct_features(i, frame_num, original_lane):
        '''
        Construct all the features for the RNN to train with consistent vehicle-centric coordinates.

        CORRECTED feature list (18 features total, indices 0-17):
        0. Existence of left lane
        1. Existence of right lane
        2. Difference of the ego car's Y position and the lane center: ΔY
        3. Ego car's Y velocity: Vy
        4. Ego car's Y acceleration: Ay
        5. Ego car's X position: X
        6. Ego car's Y position: Y
        7. Ego car's X velocity: Vx
        8. Ego car's X acceleration: Ax
        9. Ego car type: T
        10. Distance to preceding car: Dp
        11. Distance to following car: Df
        12. Distance to left preceding car: Dlp
        13. Distance to left alongside car: Dla
        14. Distance to left following car: Dlf
        15. Distance to right preceding car: Drp
        16. Distance to right alongside car: Dra
        17. Distance to right following car: Drf
        '''
        # Initialize feature dictionary
        cur_feature = {}

        # Features 0-1: Lane existence (unchanged)
        cur_feature["left_lane_exist"], cur_feature["right_lane_exist"] = determine_lane_exist(original_lane)

        # Get corrected coordinates
        transformed_coords = transform_coordinates_to_vehicle_centric(i, frame_num, original_lane)

        # Get car center with corrected Y
        car_height = tracks_meta[i][HEIGHT]
        car_center_y = transformed_coords['y'] + car_height / 2

        # Get corrected lane center
        lane_center_vehicle = get_corrected_lane_center(original_lane, lanes_info, lane_width)
        cur_feature["delta_y"] = car_center_y - lane_center_vehicle

        # Features 3-8: Vehicle state (corrected coordinates)
        cur_feature["y_velocity"] = transformed_coords['y_velocity']
        cur_feature["y_acceleration"] = transformed_coords['y_acceleration']
        cur_feature["x_position"] = transformed_coords['x']
        cur_feature["y_position"] = transformed_coords['y']
        cur_feature["x_velocity"] = transformed_coords['x_velocity']
        cur_feature["x_acceleration"] = transformed_coords['x_acceleration']

        # Feature 9: Car type (unchanged)
        cur_feature["car_type"] = 1 if tracks_meta[i][CLASS] == "Car" else -1

        # Features 10-17: Distance calculations (unchanged)
        def calculate_distance(target_car_id):
            unvalid_alter = 250
            if target_car_id != 0:
                target_frame = tracks_meta[i][INITIAL_FRAME] + frame_num - tracks_meta[target_car_id][INITIAL_FRAME]
                target_transformed = transform_coordinates_to_vehicle_centric(target_car_id, target_frame,
                                                                              original_lane)
                distance = abs(transformed_coords['x'] - target_transformed['x'])
                return distance if distance >= 0 else unvalid_alter
            else:
                return unvalid_alter

        cur_feature["preceding_distance"] = calculate_distance(tracks_csv[i][PRECEDING_ID][frame_num])
        cur_feature["following_distance"] = calculate_distance(tracks_csv[i][FOLLOWING_ID][frame_num])
        cur_feature["left_preceding_distance"] = calculate_distance(tracks_csv[i][LEFT_PRECEDING_ID][frame_num])
        cur_feature["left_alongside_distance"] = calculate_distance(tracks_csv[i][LEFT_ALONGSIDE_ID][frame_num])
        cur_feature["left_following_distance"] = calculate_distance(tracks_csv[i][LEFT_FOLLOWING_ID][frame_num])
        cur_feature["right_preceding_distance"] = calculate_distance(tracks_csv[i][RIGHT_PRECEDING_ID][frame_num])
        cur_feature["right_alongside_distance"] = calculate_distance(tracks_csv[i][RIGHT_ALONGSIDE_ID][frame_num])
        cur_feature["right_following_distance"] = calculate_distance(tracks_csv[i][RIGHT_FOLLOWING_ID][frame_num])

        return tuple(cur_feature.values())

    def detect_lane_change(lane_center, cur_y, lane_width, car_height):
        delta_y = abs(lane_center - cur_y)
        relative_diff = delta_y / car_height
        if (relative_diff < 0.5):
            return True
        else:
            return False

    def determine_change_direction(ori_laneId, new_laneId):
        '''
        return 1 upon left change
        return 2 upon right change
        '''
        if lane_num == 4:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 6 and new_laneId == 5):
                return 1
            else:
                return 2
        else:
            # left:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 4 and new_laneId == 5) \
                    or (ori_laneId == 3 and new_laneId == 4) or (ori_laneId == 7 and new_laneId == 6) \
                    or (ori_laneId == 8 and new_laneId == 7) or (ori_laneId == 9 and new_laneId == 8):
                return 1
            else:
                return 2

    # Rest of your code remains the same...
    # [The trajectory extraction logic stays identical]

    # list of list of features
    result = []

    for i in lane_changing_ids:
        # for each car:
        last_boundary = 0
        # list of (starting index, ending index, direction)
        changing_tuple_list = []
        # 1. determine the frame we want to use
        for frame_num in range(1, len(tracks_csv[i][FRAME])):
            if tracks_csv[i][LANE_ID][frame_num] != tracks_csv[i][LANE_ID][frame_num - 1]:
                original_lane = tracks_csv[i][LANE_ID][frame_num - 1]
                new_lane = tracks_csv[i][LANE_ID][frame_num]
                direction = determine_change_direction(original_lane, new_lane)
                # calculate the starting frame
                crossing_frame = frame_num - 1
                while crossing_frame > last_boundary:
                    if detect_lane_change(lanes_info[original_lane], tracks_csv[i][Y][crossing_frame], lane_width,
                                          tracks_meta[i][HEIGHT]):
                        break
                    crossing_frame -= 1
                # calculate the starting and ending frame
                if INCLUDE_AFTER_CROSSING:
                    starting_point = crossing_frame - TRAJECTORY_LENGTH + FRAMES_AFTER_CROSSING
                    ending_point = crossing_frame + FRAMES_AFTER_CROSSING
                else:
                    starting_point = crossing_frame - TRAJECTORY_LENGTH
                    ending_point = crossing_frame
                if starting_point > last_boundary:
                    changing_tuple_list.append(
                        (starting_point, ending_point, direction))
                last_boundary = frame_num

        # add those frames' features
        # Inside the loop that processes each lane change instance
        for pair in changing_tuple_list:
            # for each lane change instance
            cur_change = []
            start_idx = pair[0]
            end_idx = pair[1]
            direction = []
            original_lane = tracks_csv[i][LANE_ID][start_idx]

            # continue for out of boundary cases
            if original_lane not in lanes_info:
                continue

            # Check if end_idx is beyond available frames for this vehicle
            end_idx = min(end_idx, len(tracks_csv[i][FRAME]) - 1)

            for frame_num in range(start_idx, end_idx):
                # construct the object
                cur_change.append(construct_features(
                    i, frame_num, original_lane))
                direction.append(pair[2])
            # add to the result
            result.append((cur_change, direction))

    change_num = len(result)

    # Calculate desired LK sample size (3x the LC trajectories)
    desired_lk_size = len(result) * 3

    if len(lane_keeping_ids) > desired_lk_size:
        # Sample 3x lane changes
        lane_keeping_ids = random.sample(lane_keeping_ids, desired_lk_size)
    else:
        # Use all available lane keeping vehicles (less than 3x but that's all we have)
        print(f"Warning: Only {len(lane_keeping_ids)} LK vehicles available, wanted {desired_lk_size}")

    for i in lane_keeping_ids:
        cur_change = []
        original_lane = tracks_csv[i][LANE_ID][0]
        fail = False
        direction = []
        for frame_num in range(1, TRAJECTORY_LENGTH + 1):
            try:
                cur_change.append(construct_features(
                    i, frame_num, original_lane))
                direction.append(0)

            except:
                # handle exception where the total frame is less than TRAJECTORY_LENGTH
                fail = True
                break
        if not fail:
            result.append((cur_change, direction))

    return result, change_num