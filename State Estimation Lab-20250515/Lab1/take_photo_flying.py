from djitellopy import Tello
import cv2

tello = Tello()
tello.connect()
tello.streamon()

cv2.namedWindow("Tello Stream")
photo_id = 0
SPEED = 50  # cm

KEY_BINDINGS = {
    'j': 'Takeoff',
    'l': 'Land',
    'w': 'Move Forward',
    's': 'Move Backward',
    'a': 'Move Left',
    'd': 'Move Right',
    'r': 'Move Up',
    'f': 'Move Down',
    'e': 'Rotate Clockwise',
    'q': 'Rotate Counter-Clockwise',
    'k': 'Take Photo',
    'h': 'Show Help',
    'x': 'Exit (with landing)',
}

def print_help():
    print("=== Tello Control Keys ===")
    for key, action in KEY_BINDINGS.items():
        print(f"'{key}' : {action}")
    print("==========================")

print_help()

try:
    while True:
        frame = tello.get_frame_read().frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Tello Stream", rgb_frame)
        
        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key) if key != 255 else ''

        if key_char == 'x':
            print("Landing before exit...")
            tello.land()
            break
        elif key_char == 'j':
            tello.takeoff()
        elif key_char == 'l':
            tello.land()
        elif key_char == 'w':
            tello.move_forward(SPEED)
        elif key_char == 's':
            tello.move_back(SPEED)
        elif key_char == 'a':
            tello.move_left(SPEED)
        elif key_char == 'd':
            tello.move_right(SPEED)
        elif key_char == 'r':
            tello.move_up(SPEED)
        elif key_char == 'f':
            tello.move_down(SPEED)
        elif key_char == 'e':
            tello.rotate_clockwise(45)
        elif key_char == 'q':
            tello.rotate_counter_clockwise(45)
        elif key_char == 'k':
            filename = f"photo_{photo_id:03d}.jpg"
            cv2.imwrite(filename, rgb_frame)
            print(f"Saved {filename}")
            photo_id += 1
        elif key_char == 'h':
            print_help()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()
