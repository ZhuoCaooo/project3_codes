from read_data import *
import random


''''IMPORTANT: in this script, all the trajectories after extraction are moving into positive X-axis direction. And negative
Y-axis movment means Right change, positives meand left lane change'''
# HighD frame_rate= 25 Hz
FRAME_TAKEN = 200  # 8 seconds total (4 before + 4 after boundary crossing)
FRAME_BEFORE = 100  # 4 seconds before boundary crossing
FRAME_AFTER = 100   # 4 seconds after boundary crossing


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
        if(tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
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

    def construct_features(i, frame_num, original_lane):
        '''
        Construct all the features for extracted trajectories:
        Here is the list:
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
        going = 0  # 1 left, 2 right
        if lane_num == 4:
            if original_lane == 2 or original_lane == 3:
                going = 1
            else:
                going = 2
        else:
            if original_lane == 2 or original_lane == 3 or original_lane == 4 or original_lane == 5:
                going = 1
            else:
                going = 2
        cur_feature = {}
        cur_feature["left_lane_exist"], cur_feature["right_lane_exist"] = determine_lane_exist(
            original_lane)

        # We need to consider the fact that right/left are different for top/bottom lanes.
        # top lanes are going left      <----
        # bottom lanes are going right  ---->
        # left -> positive, right -> negative
        car_center = tracks_csv[i][Y][frame_num] + tracks_meta[i][HEIGHT] / 2
        if going == 1:
            cur_feature["delta_y"] = car_center - \
                lanes_info[original_lane] - lane_width/2  # up
            cur_feature["y_velocity"] = tracks_csv[i][Y_VELOCITY][frame_num]
            cur_feature["y_acceleration"] =  \
                tracks_csv[i][Y_ACCELERATION][frame_num]
            # the position of the ego vehicle:
            cur_feature["x_position"] = -tracks_csv[i][X][frame_num]
            cur_feature["y_position"] = tracks_csv[i][Y][frame_num]
            cur_feature["x_velocity"] = -tracks_csv[i][X_VELOCITY][frame_num]
            cur_feature["x_acceleration"] = -tracks_csv[i][X_ACCELERATION][frame_num]
        else:
            cur_feature["delta_y"] = lanes_info[original_lane] - \
                car_center + lane_width/2  # down
            cur_feature["y_velocity"] = -tracks_csv[i][Y_VELOCITY][frame_num]
            cur_feature["y_acceleration"] = -tracks_csv[i][Y_ACCELERATION][frame_num]
            # the position of the ego vehicle:
            cur_feature["x_position"] = tracks_csv[i][X][frame_num]
            cur_feature["y_position"] = -tracks_csv[i][Y][frame_num]
            cur_feature["x_velocity"] = tracks_csv[i][X_VELOCITY][frame_num]
            cur_feature["x_acceleration"] = tracks_csv[i][X_ACCELERATION][frame_num]

        cur_feature["car_type"] = 1 if tracks_meta[i][CLASS] == "Car" else -1


        def calculate_distance(target_car_id):
            """
            Calculate distance between target car and current car
            """
            unvalid_alter = 250  # Large distance for no vehicle
            if target_car_id != 0:
                target_frame = tracks_meta[i][INITIAL_FRAME] + \
                               frame_num - tracks_meta[target_car_id][INITIAL_FRAME]
                target_x = tracks_csv[target_car_id][X][target_frame]
                cur_x = tracks_csv[i][X][frame_num]
                distance = abs(cur_x - target_x)
                return distance
            else:
                return unvalid_alter

        # surrounding cars info
        cur_feature["preceding_ttc"] = calculate_distance(
            tracks_csv[i][PRECEDING_ID][frame_num])

        cur_feature["following_ttc"] = calculate_distance(
            tracks_csv[i][FOLLOWING_ID][frame_num])

        cur_feature["left_preceding_ttc"] = calculate_distance(
            tracks_csv[i][LEFT_PRECEDING_ID][frame_num])

        cur_feature["left_alongside_ttc"] = calculate_distance(
            tracks_csv[i][LEFT_ALONGSIDE_ID][frame_num])

        cur_feature["left_following_ttc"] = calculate_distance(
            tracks_csv[i][LEFT_FOLLOWING_ID][frame_num])

        cur_feature["right_preceding_ttc"] = calculate_distance(
            tracks_csv[i][RIGHT_PRECEDING_ID][frame_num])

        cur_feature["right_alongside_ttc"] = calculate_distance(
            tracks_csv[i][RIGHT_ALONGSIDE_ID][frame_num])

        cur_feature["right_following_ttc"] = calculate_distance(
            tracks_csv[i][RIGHT_FOLLOWING_ID][frame_num])

        ret = tuple(cur_feature.values())
        return ret

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

    # list of list of features
    result = []

    for i in lane_changing_ids:
        # for each car:
        last_boundary = 0
        # list of (starting index, ending index, direction)
        changing_tuple_list = []

        # 1. Find lane boundary crossing points
        for frame_num in range(1, len(tracks_csv[i][FRAME])):
            if tracks_csv[i][LANE_ID][frame_num] != tracks_csv[i][LANE_ID][frame_num - 1]:
                # Lane ID changed - this is our boundary crossing time T
                boundary_crossing_frame = frame_num
                original_lane = tracks_csv[i][LANE_ID][frame_num - 1]
                new_lane = tracks_csv[i][LANE_ID][frame_num]
                direction = determine_change_direction(original_lane, new_lane)

                # Extract [T-4s, T+4s] = [T-100, T+100]
                starting_point = boundary_crossing_frame - FRAME_BEFORE  # T-100
                ending_point = boundary_crossing_frame + FRAME_AFTER  # T+100

                # Check if we have enough data before and after
                if (starting_point > last_boundary and
                        ending_point < len(tracks_csv[i][FRAME]) and
                        starting_point >= 0):
                    changing_tuple_list.append(
                        (starting_point, ending_point, direction, boundary_crossing_frame))

                last_boundary = ending_point  # Avoid overlapping windows

        # Process each lane change instance
        for pair in changing_tuple_list:
            cur_change = []
            start_idx = pair[0]  # T-100
            end_idx = pair[1]  # T+100
            direction = []

            original_lane = tracks_csv[i][LANE_ID][start_idx]

            # Skip if original lane not in lanes_info
            if original_lane not in lanes_info:
                continue

            for frame_num in range(start_idx, end_idx):
                cur_change.append(construct_features(i, frame_num, original_lane))
                direction.append(pair[2])

            # Consistent structure: just features and directions
            result.append((cur_change, direction))

    change_num = len(result)

    # Define the desired number of lane-keeping samples
    lk_sample_size = 2 * change_num

    # Check if you have enough lane-keeping cars to sample from
    if len(lane_keeping_ids) >= lk_sample_size:
        # Sample twice the number of lane-changing events
        lane_keeping_ids = random.sample(lane_keeping_ids, lk_sample_size)

    for i in lane_keeping_ids:
        cur_change = []
        original_lane = tracks_csv[i][LANE_ID][0]
        fail = False
        direction = []
        for frame_num in range(1, FRAME_TAKEN+1):
            try:
                cur_change.append(construct_features(
                    i, frame_num, original_lane))
                direction.append(0)

            except:
                # handle exception where the total frame is less than FRAME_TAKEN
                fail = True
                break
        if not fail:
            result.append((cur_change, direction))

    return result, change_num
