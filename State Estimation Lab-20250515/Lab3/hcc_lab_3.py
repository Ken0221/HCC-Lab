import numpy as np
import time
from djitellopy import Tello
from pupil_apriltags import Detector
import cv2
import matplotlib.pyplot as plt

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
        self.R = np.eye(3) * 1e-2  # 過程雜訊 -> 如果我們更相信觀測值，可以增加這個值
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

        for tag in tags:
            if tag.pose_t is not None:
                t = tag.pose_t.flatten()  # (x, y, z) relative to camera
                print(f"[Tag {tag.tag_id}] x={t[0]:.2f} y={t[1]:.2f} z={t[2]:.2f}")

                # Draw tag on image
                corners = np.int32(tag.corners)
                center = tuple(np.int32(tag.center))
                cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {tag.tag_id}", (center[0]+10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                tag_found = True
                break

        # Show live feed
        cv2.imshow("Tello Stream with AprilTag", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    return t 
    
def tello_command(movement_request):
    dp = [0.0, 0.0, 0.0]
    cmd, value = movement_request
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
    

if __name__ == '__main__':
    movement_sequence = [
    ("forward", 30),
    ("forward", 20),
    ("forward", 20),
    ("left", 20),
    ("forward", 30),
    ("right", 20),
    ("forward", 30),
    ("right", 20),
    ("back", 30),
    ("back", 20),
    ("back", 20)
    ]

    # Camera calibration (replace with your values if needed)
    camera_params = [955.91327988, 935.28228658, 357.43963893, 365.75928463] # fx, fy, cx, cy UAV01
    # camera_params = [955.91327988, 935.28228658, 450.43963893, 330.75928463] # fx, fy, cx, cy UAV01
    # camera_params = [947.562187484957, 938.150409141504, 405.68710563344047, 393.41441745089753]
    # camera_params = [971.545, 951.018, 353.092, 380.145]
    tag_size = 0.166 # meters

    # AprilTag detector
    at_word = np.array([0.0, 3.0, 0.0]) #x, y, z
    at_detector = Detector(families='tag36h11')

    # Kalman Filter
    KF = KalmanFilter()

    # Initialize
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    frame_read = tello.get_frame_read()
    frame = None
    while frame is None:
        frame = frame_read.frame
    # time.sleep(3)

    tello.takeoff()
    time.sleep(2)

    drone_wpose_ct = np.array([0.0, 0.0, 0.0])
    d_wposes_ct = []
    d_wposes_at = []
    d_wposes_kf = []
    for i in range(len(movement_sequence)):
        dp = tello_command(movement_sequence[i])
        drone_wpose_ct += dp
        at_pose = Detect_AprilTag(frame_read)
        # at_pose = at_simulation[i]
        drone_wpose_at = [at_word[0] - at_pose[0], at_word[1] - at_pose[2], at_word[2] + at_pose[1]]
        KF.predict(np.expand_dims(dp, axis=1))
        drone_wpose_kf = KF.update(np.expand_dims(drone_wpose_at, axis=1))
        d_wposes_ct.append(drone_wpose_ct.copy())
        d_wposes_at.append(drone_wpose_at)
        d_wposes_kf.append(drone_wpose_kf.flatten())
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            print("Landing...")
            tello.land()
            break
    
    np.savetxt('poses_ct.txt', d_wposes_ct)
    np.savetxt('poses_at.txt', d_wposes_at)
    np.savetxt('poses_kf.txt', d_wposes_kf)

    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()

# === Plot 2D trajectory ===
    plot_trajectory(d_wposes_ct, d_wposes_at, d_wposes_kf)