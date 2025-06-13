import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

min_step = 20
max_step = 500
direction_num = [["right", "left"], ["forward", "back"], ["down", "up"]]
        
known_tags_pos_dict = {
    100: np.array([0.0, 0.0, 0.0]),  # x, y, z in meters
    101: np.array([0.8, 0.0, 0.0]),
    102: np.array([1.3, 0.0, 0.0]),
    103: np.array([1.55, 1.9, 0.0]),
    104: np.array([1.05, 3.0, 0.0]),
    105: np.array([-1.5, 1.1, 0.0]),
    106: np.array([-1.2, 0.0, 0.0]),
    107: np.array([-0.7, 0.0, 0.0]),
}
        
dest_pos_dict = {
    1: np.array([0.75, 0.75, 0.0]),
    2: np.array([0.0, 0.75, 0.0]),
    3: np.array([-0.75, 0.75, 0.0]),
    4: np.array([0.75, 1.5, 0.0]),
    5: np.array([0.0, 1.5, 0.0]),
    6: np.array([-0.75, 1.5, 0.0]),
    7: np.array([0.75, 2.25, 0.0]),
    8: np.array([0.0, 2.25, 0.0]),
    9: np.array([-0.75, 2.25, 0.0]),
}

professor_dest_dict = {
    0: 2, # klw
    1: 3, # shh
    2: 8, # wjc
    3: 4, # wlj
}

professor_name_dict = {
    0: "klw",
    1: "shh",
    2: "wjc",
    3: "wlj",
}

model = YOLO("/Users/ken/Desktop/113-2/HCC Lab/final/runs/train/weights/best.pt")

    
drone_init_pos = np.array([0.0, 4.05, -0.4])  # x, y, z in meters
# apriltag_pos = np.array([0.0, 3.0, 0.0])  # x, y, z in meters
# center_tag_id = 0  # The tag ID of the center tag

# Camera calibration (replace with your values if needed)
# camera_params = [955.91327988, 900.28228658, 357.43963893, 250.75928463] # fx, fy, cx, cy UAV01
camera_params = [958.91327988, 938.28228658, 487.43963893, 365.75928463] # fx, fy, cx, cy UAV01
# camera_params = [682.6036696139109, 656.8892545247161, 565.128879457693, 463.17204457792616]
tag_size = 0.166 # meters
# tag_size = 0.06

# AprilTag detector
# at_word = apriltag_pos
at_detector = Detector(families='tag36h11')

drone_wpose_ct = drone_init_pos
d_wposes_ct = [[0.0, 4.05, -0.4]]
d_wposes_at = []
d_wposes_kf = []

class KalmanFilter:
    def __init__(self):
        """
        Parameters:
        -----------
        u : Control input (m x 1)
        z : Observation (k x 1)
        A : State transition matrix (n x n)
        B : Control input model (n x m)
        C : Observation matrix (k x n)
        R : Process noise covariance matrix (n x n)
        Q : Measurement noise covariance matrix (k x k)
        mu : state estimate (n x 1)
        Sigma : state covariance (n x n)
        """
        # TODO_1
        self.A = np.eye(3)  # 狀態轉移矩陣
        self.B = np.eye(3)  # 控制輸入矩陣
        self.C = np.eye(3)  # 觀測模型
        self.R = np.eye(3) * 3.5e-1  # 過程雜訊 -> 如果我們更相信觀測值，可以增加這個值
        self.Q = np.eye(3) * 1  # 觀測雜訊 -> 如果我們更相信控制輸入，可以增加這個值
        self.mu = np.zeros((3, 1))  # 初始狀態估計 (x, y, z)
        self.Sigma = np.eye(3) * 1e-2  # 初始協方差 -> 這個值越小，表示我們對初始狀態的信心越高

    def predict(self, u):
        # TODO_2
        self.mu = self.A @ self.mu + self.B @ u
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.R

    def update(self, z):
        # TODO_3
        K = self.Sigma @ self.C.T @ np.linalg.inv(self.C @ self.Sigma @ self.C.T + self.Q)
        self.mu = self.mu + K @ (z - self.C @ self.mu)
        self.Sigma = (np.eye(self.Sigma.shape[0]) - K @ self.C) @ self.Sigma
        return self.mu

    def get_state(self):
        return self.mu, self.Sigma

def Detect_AprilTag(frame_read):
    tag_found = False
    while not tag_found:
        frame = frame_read.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect tags
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
        # print(f"Detected {len(tags)} tags: ")
        # print("tag_info: ", tags)
        
        t = []
        return_id = []
        for tag in tags:
            if tag.pose_t is not None:
                # t = tag.pose_t.flatten()  # (x, y, z) relative to camera
                t.append(tag.pose_t.flatten())  # (x, y, z) relative to camera
                # print(f"[Tag {tag.tag_id}] x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}")
                print(f"[Tag {tag.tag_id}] x={t[-1][0]:.2f} y={t[-1][1]:.2f} z={t[-1][2]:.2f}")
                
                # return_id = tag.tag_id
                return_id.append(tag.tag_id)

                # Draw tag on image
                corners = np.int32(tag.corners)
                center = tuple(np.int32(tag.center))
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {tag.tag_id}", (center[0]+10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    
                tag_found = True
                # break

        # Show live feed
        cv2.imshow("Tello Stream with AprilTag", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    return t, return_id
    
def tello_command(movement_request):
    dp = [0.0, 0.0, 0.0]
    cmd, value = movement_request
    value = int(value)
    if cmd == "forward":
        tello.move_forward(value)
        dp = [0.0, value, 0.0]
    elif cmd == "back":
        tello.move_back(value)
        dp = [0.0, -value, 0.0]
    elif cmd == "right":
        tello.move_right(value)
        dp = [value, 0.0, 0.0]
    elif cmd == "left":
        tello.move_left(value)
        dp = [-value, 0.0, 0.0]
    elif cmd == "up":
        tello.move_up(value)
        dp = [0.0, 0.0, value]
    elif cmd == "down":
        tello.move_down(value)
        dp = [0.0, 0.0, -value]
    elif cmd == "cw":
        tello.rotate_clockwise(value)
        dp = [0.0, 0.0, 0.0]
    elif cmd == "ccw":
        tello.rotate_counter_clockwise(value)
        dp = [0.0, 0.0, 0.0]
    time.sleep(1)

    return np.array(dp) * 0.01

def get_current_position_kf():
    """
    Function to get the current position of the drone.
    This is a placeholder function and should be replaced with actual position retrieval logic.
    Returns:
    np.array
        The current position in the format [x, y, z].
    """
    # Replace with actual position retrieval logic
    get_id = False
    drone_wpose_at_avg = np.zeros(3)
    drone_wpose_kf_avg = np.zeros((3, 1)) 
    id_cnt = 0
    while(not get_id):
        at_pose_arr, return_id_arr = Detect_AprilTag(frame_read)
        # print("at_pose: ", at_pose)
        for at_pose, return_id in zip(at_pose_arr, return_id_arr):
            if return_id in known_tags_pos_dict:
                id_cnt += 1
                tag_pos = known_tags_pos_dict[return_id]
                # print("tag_pos: ", tag_pos)
                drone_wpose_at = [tag_pos[0] + at_pose[0], tag_pos[1] + at_pose[2], tag_pos[2] + at_pose[1]] # 有改
                drone_wpose_kf = KF.update(np.expand_dims(drone_wpose_at, axis=1))
                drone_wpose_at_avg += drone_wpose_at
                drone_wpose_kf_avg += drone_wpose_kf
                
                get_id = True
            else:
                print("key: ", return_id)
                print("No key in dict!")
    
    drone_wpose_at_avg /= id_cnt
    # drone_wpose_at_avg[1] -= 0.15
    drone_wpose_kf_avg /= id_cnt
    d_wposes_at.append(drone_wpose_at_avg)
    d_wposes_kf.append(drone_wpose_kf_avg.flatten())
    print(f"Drone Position (AT): {drone_wpose_at_avg}")
    print(f"Drone Position (CT): {drone_wpose_ct}")
    print(f"Drone Position (KF): {drone_wpose_kf_avg.flatten()}")
    
    return drone_wpose_kf.flatten()

def fly_to(current_pos, dest_pos):
    """
    Function to fly the drone to a specific destination position.
    Parameters:
    dest_pos : np.array
        The destination position in the format [x, y, z].
    """
    # Get the current position of the drone
    # current_pos = get_current_position_kf()
    print(f"Destination Position: {dest_pos}")
    # Calculate the movement vector
    movement_path = ((dest_pos - current_pos) * 100).astype(int)
    print(f"Movement Vector: {movement_path}")
    # Calculate the distance from the destination tag
    for i in range(len(movement_path)):
        if abs(movement_path[i]) < min_step / 2:
            movement_path[i] = 0
        if abs(movement_path[i]) < min_step and abs(movement_path[i]) > 0:
            movement_path[i] = min_step if movement_path[i] > 0 else -min_step
    print("movement_path: ", movement_path)
    
    movement_sequence = []
    for i in [1, 0]: # only x, y
        while(abs(movement_path[i]) >= min_step):
            # print(f"movement_path[{i}]: {movement_path[i]}")
            if abs(movement_path[i]) - max_step >= min_step:
                move_step = max_step
            elif abs(movement_path[i]) < max_step:
                move_step = abs(movement_path[i])
            else:
                move_step = (abs(movement_path[i]) / 2).astype(int)
                
            if movement_path[i] > 0:
                movement_sequence.append((direction_num[i][1], move_step))
                movement_path[i] -= move_step

            elif movement_path[i] < 0:
                movement_sequence.append((direction_num[i][0], move_step))
                movement_path[i] += move_step


    print("movement_sequence: ", movement_sequence)  
    
    global drone_wpose_ct
    for i in range(len(movement_sequence)):
        dp = tello_command(movement_sequence[i])
        if not (movement_sequence[i][0] == "up" or movement_sequence[i][0] == "down"):
            dp = -dp
        KF.predict(np.expand_dims(dp, axis=1))
        drone_wpose_ct += dp
        d_wposes_ct.append(drone_wpose_ct.copy())
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            print("Landing...")
            tello.land()
            break
    
    actual_pos = get_current_position_kf()
    print(f"Arrive actual position: {actual_pos}")
    
    return actual_pos
    

def plot_trajectory(control_poses, tag_pose, kalmanfilter_pose):
    poses_ct = np.array(control_poses)
    poses_at = np.array(tag_pose)
    poses_kf = np.array(kalmanfilter_pose)
    plt.figure(figsize=(8, 6))
    plt.plot(poses_ct[:, 0], poses_ct[:, 1], 'ko--', label='Motion Model')
    plt.plot(poses_at[:, 0], poses_at[:, 1], 'rx-', label='AprilTag')
    plt.plot(poses_kf[:, 0], poses_kf[:, 1], 'b^-', label='Kalman Filter')
    plt.title("2D Trajectory Tracking")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    
def Detect_target_AprilTag(frame_read, target, unknown_tag_id):
    target_found = False
    unknown_found = False
    tag_found = False
    count = 0
    while not tag_found:
        count = count + 1
        if count > 5:
            tag_found = 1
            id = 10000
            target_pos = [0,0,0]
            unknown_pos = [0,0,0]
            break
        time.sleep(0.2)
        target_found = False
        unknown_found = False
        frame = frame_read.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        for tag in tags:
            if tag.pose_t is not None:
                # print(tag.tag_id)
                if tag.tag_id == target:
                    target_found = True
                    target_pos = tag.pose_t.flatten()  # (x, y, z) relative to camera
        for tag in tags:
            if tag.pose_t is not None:
                # print(tag.tag_id)
                if tag.tag_id == unknown_tag_id:
                    unknown_found = True
                    unknown_pos = tag.pose_t.flatten()  # (x, y, z) relative to camera
                    id = tag.tag_id
        tag_found = target_found and unknown_found
        # Show live feed
        cv2.imshow("Tello Stream with AprilTag", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    return unknown_pos, id, target_pos

def find_unknown_tag(frame_read):
    find_tag_dict = {}
    unknown_tag_id_left = 201
    unknown_tag_id_right = 202
    unknown_tag_id_front = 200
    unknown_tag_id_back = -1
    
    print("ready to find")
    # actual_pos = get_current_position_kf()
    # actual_pos = fly_to(actual_pos, dest_pos_dict[8])
    #unknown_tag,id = Detect_unknown_Tag(frame_read)
    
    # front
    ref_tag_id = 100
    unknown_tag_id = unknown_tag_id_front
    unknown_tag,id,target_tag = Detect_target_AprilTag(frame_read, ref_tag_id, unknown_tag_id)
    if id == 10000:
        print("     not in front")
    else :
        print(" Ans Tag ",id," at [95, 0, 0]")
        print("     Tag ",id," at [",(target_tag[0] - unknown_tag[0] )* 100,", ",0,", ",unknown_tag[2], "]")
        find_tag_dict[id] = np.array([(target_tag[0] - unknown_tag[0]) * 100, 0, unknown_tag[2]])
        # known_tags_pos_dict[id] = np.array([(target_tag[0] - unknown_tag[0]) * 100, 0, unknown_tag[2]])
            
    rotate_for_side = 110
    rotate_for_botton = 50
    rotate_for_motify_back = 10
    rotate_modify_time_back = 4
    rotate_for_motify_side = 5
    rotate_modify_time_side = 3
    
    if unknown_tag_id_back == -1:
        rotate_modify_time_back = 0

    # left
    if unknown_tag_id_left != -1:
        actual_pos = get_current_position_kf()
        actual_pos = fly_to(actual_pos, [dest_pos_dict[3][0], dest_pos_dict[3][1] + 0.15, dest_pos_dict[3][2]])
        
        ref_tag_id = 103
        unknown_tag_id = unknown_tag_id_left
        modify_time_side = 0
        tello.rotate_counter_clockwise(rotate_for_side)
        time.sleep(1)
        frame_read = tello.get_frame_read()
        print("     ready to find second unknown")
        for i in range(rotate_modify_time_side):
            unknown_tag,id,target_tag = Detect_target_AprilTag(frame_read,ref_tag_id, unknown_tag_id)
            if id == 10000:
                print("     not in left")
                tello.rotate_counter_clockwise(rotate_for_motify_side)
                time.sleep(1)
                modify_time_side += 1
            else :
                print(" Ans Tag ",id," at [155, 109, 0]")
                print("     Tag ",id," at [",155,", ",190 + (target_tag[0] - unknown_tag[0])*100,", ", unknown_tag[2], "]")
                find_tag_dict[id] = np.array([155, 190 + (target_tag[0] - unknown_tag[0]) * 100, unknown_tag[2]])
                break
        
        rotate_for_botton_1 = rotate_for_botton - modify_time_side * rotate_for_motify_side
        tello.rotate_counter_clockwise(rotate_for_botton_1)
        time.sleep(1)

        modify_time_back = 0
        if unknown_tag_id_back != -1:
            # back
            Is_find_back = False
            ref_tag_id = 104
            unknown_tag_id = unknown_tag_id_back
            
            frame_read = tello.get_frame_read()
            print("     ready to find fourth unknown")
            for i in range(rotate_modify_time_back):
                unknown_tag,id,target_tag = Detect_target_AprilTag(frame_read, ref_tag_id, unknown_tag_id)
                if id == 10000:
                    print("     not in back")
                    tello.rotate_counter_clockwise(rotate_for_motify_back)
                    time.sleep(1)
                    modify_time_back += 1
                else :
                    print(" Ans Tag ",id," at [-1, -1, -1]")
                    print("     Tag ",id," at [",105 -  (target_tag[0] - unknown_tag[0] )* 100,", ",300,", ",unknown_tag[2], "]")
                    find_tag_dict[id] = np.array([105 - (target_tag[0] - unknown_tag[0]) * 100, 300, unknown_tag[2]])
                    Is_find_back = True
                    break
        else:
            rotate_for_botton_1 = 0
        
        tello.rotate_clockwise(rotate_for_side + modify_time_side * rotate_for_motify_side + rotate_for_botton_1 + modify_time_back * rotate_for_motify_back)

        # tello.rotate_counter_clockwise(5) # adjust the direction just in case
        # time.sleep(1)
    
    # right
    if unknown_tag_id_right != -1:
        actual_pos = get_current_position_kf()
        actual_pos = fly_to(actual_pos, [dest_pos_dict[1][0] - 0.1, dest_pos_dict[1][1] + 0.2, dest_pos_dict[1][2]])
        tello.rotate_clockwise(rotate_for_side)
        time.sleep(1)

        ref_tag_id = 105
        unknown_tag_id = unknown_tag_id_right
        modify_time_side = 0
        frame_read = tello.get_frame_read()
        print("     ready to find third unknown")
        for i in range(3):
            unknown_tag,id,target_tag = Detect_target_AprilTag(frame_read, ref_tag_id, unknown_tag_id)
            if id == 10000:
                print("     not in right")
                tello.rotate_clockwise(rotate_for_motify_side)
                time.sleep(1)
                modify_time_side += 1
            else :
                print(" Ans Tag ",id," at [-150, 206, 0]")
                print("     Tag ",id," at [",-150,", ",110 - (target_tag[0] - unknown_tag[0])*100,", ",unknown_tag[2], "]")
                find_tag_dict[id] = np.array([-150, 110 - (target_tag[0] - unknown_tag[0]) * 100, unknown_tag[2]])
                break
        
        rotate_for_botton_2 = rotate_for_botton - modify_time_side * rotate_for_motify_side

        modify_time_back = 0
        if unknown_tag_id_back != -1:
            # back
            ref_tag_id = 104
            unknown_tag_id = unknown_tag_id_back
            
            rotate_total = 0
            if not Is_find_back:
                tello.rotate_clockwise(rotate_for_botton_2)
                time.sleep(1)
                rotate_total += rotate_for_botton_2
                frame_read = tello.get_frame_read()
                print("     ready to find fourth unknown")
                for i in range(rotate_modify_time_back):
                    unknown_tag,id,target_tag = Detect_target_AprilTag(frame_read, ref_tag_id, unknown_tag_id)
                    if id == 10000:
                        print("     not in back")
                        tello.rotate_clockwise(rotate_for_motify_back)
                        modify_time_back += 1
                    else :
                        print(" Ans Tag ",id," at [-1, -1, -1]")
                        print("     Tag ",id," at [",105 -  (target_tag[0] - unknown_tag[0] )* 100,", ",300,", ",unknown_tag[2], "]")
                        find_tag_dict[id] = np.array([105 - (target_tag[0] - unknown_tag[0]) * 100, 300, unknown_tag[2]])
                        break
        else:
            rotate_for_motify_back = 0
            
        rotate_total += rotate_for_side + modify_time_side * rotate_for_motify_side + modify_time_back * rotate_for_motify_back
        tello.rotate_counter_clockwise(rotate_total)
        # tello.rotate_counter_clockwise(5) # adjust the direction just in case

        # time.sleep(1)   
    
    return find_tag_dict 

def detect_professor(frame_read):
    professor_cnt = [0, 0, 0, 0]
    times = 5
    correct_time = 4
    for i in range(times):
        # Step 1: 擷取影像（frame 是 numpy array）
        frame = frame_read.frame  # 這才是 numpy 陣列
        # Step 2: 將 BGR 轉成 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 3: 傳入模型做預測
        results = model.predict(source=frame_rgb, conf=0.7, iou=0.7)[0]
        boxes = results.boxes
        
        #confidence = 1     # confidence score
        #class_name = detection[0].names[int(detection[0].cls)]  # predicted class name

        cv2.imshow("Tello Stream with AprilTag", frame_rgb)
        cv2.imshow("Tello Stream with AprilTag", frame)
        key = cv2.waitKey(1)
        if len(boxes) > 1:
            continue
        for box in boxes:
            print(int(box.cls))
            professor_cnt[int(box.cls)] += 1
    max_cnt = max(professor_cnt)
    index = professor_cnt.index(max_cnt)
    if max_cnt < correct_time:
        return None
    
    return index

if __name__ == '__main__':
    # d_wposes_ct.append(drone_init_pos)
    # print(d_wposes_ct)
    
    # Kalman Filter
    KF = KalmanFilter()
    KF.predict(np.expand_dims(drone_init_pos, axis=1))
    
    # Initialize
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_read = tello.get_frame_read()
    frame = None
    while frame is None:
        frame = frame_read.frame
    time.sleep(3)
    
    tello.takeoff()
    time.sleep(1)
    
    # while(1):
        # at_pose, return_id = Detect_AprilTag(frame_read)
        # print("return_id: ", return_id)
        # get_current_position_kf()
        
    professor_pred = None
    while professor_pred is None:
        professor_pred = detect_professor(frame_read)
    print("professor_pred: ", professor_name_dict[professor_pred])
    dest_id = professor_dest_dict[professor_pred]
    print("dest_id: ", dest_id)
        
    # exit(1)
        
    # dest_id = 4 # for test
            
    tello_command(("up", 50))
    time.sleep(1)
        
    find_tag_dict = find_unknown_tag(frame_read)

    # Get the position of destination
    dest_pos = np.array(dest_pos_dict[dest_id])
    if dest_id == 3: #s
        dest_pos[1] -= 0.1
    if dest_id == 2: #k
        dest_pos[0] += 0.1
        dest_pos[1] += 0.1
    if dest_id == 4: #l
        dest_pos[1] += 0.05
        # dest_pos[0] += 0.1
    if dest_id == 6:
        dest_pos[1] += 0.05
    if dest_id == 7:
        dest_pos[1] += 0.05
    if dest_id == 8: #j
        dest_pos[1] -= 0.1
    if dest_id == 9:
        dest_pos[1] -= 0.05
        
    print("dest_pos: ", dest_pos)
            
    error_at = np.array([np.inf, np.inf, np.inf])  # Initialize error to a large value
    actual_pos = get_current_position_kf()
    adjust_times = 10
    while adjust_times > 0 and (np.abs(error_at[0]) >= (min_step / 2) / 100) or (np.abs(error_at[1]) >= (min_step / 2) / 100):   
        adjust_times -= 1
        # Fly to the destination
        actual_pos = fly_to(actual_pos, dest_pos)
                
        # Get error
        error_at = np.array(dest_pos) - np.array(actual_pos)
        print("error: ", error_at)
        print("")
            
    if adjust_times <= 0:
        print("Adjust times exceeded, landing...")
            
    tello.land()
        
    
            
    np.savetxt('poses_ct.txt', d_wposes_ct)
    np.savetxt('poses_at.txt', d_wposes_at)
    np.savetxt('poses_kf.txt', d_wposes_kf)
    
    # print(known_tags_pos_dict)
    print("find_tag_dict: ", find_tag_dict)

    # tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

# === Plot 2D trajectory ===
    plot_trajectory(d_wposes_ct, d_wposes_at, d_wposes_kf)