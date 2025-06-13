import time
import rclpy
from rclpy.node import Node
from tello_msgs.srv import TelloAction
from tello_msgs.msg import TelloResponse 
from geometry_msgs.msg import Twist


class TelloCommander(Node):

    def __init__(self):
        super().__init__('tello_commander')
        
        self.cli = self.create_client(TelloAction, '/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /tello_action service...')
        
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def send_command(self, command: str):
        req = TelloAction.Request()
        req.cmd = command
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Command '{command}' sent successfully.")
        else:
            self.get_logger().error(f"Failed to send command '{command}'.")

    def publish_velocity(self, rc_command):

        vy, vx, vz, v_yaw = rc_command
        
        linear_x = float(vx) / 100
        linear_y = float(vy) / 100
        linear_z = float(vz) / 100
        angular_z = float(v_yaw)

        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.linear.z = linear_z
        twist.angular.z = angular_z
        self.vel_publisher.publish(twist)
        self.get_logger().info(f"Published velocity: linear=({linear_x},{linear_y},{linear_z}) angular=({angular_z})")


def main():

    rclpy.init()
    commander = TelloCommander()

    try:
       
        commander.send_command("takeoff")
        time.sleep(5.0)
        
        rc_command = [0.0, 20.0, 0.0, 0.0]
        for _ in range(65):
            commander.publish_velocity(rc_command)
            time.sleep(0.1)
        '''
        rc_command = [0, 0, 0, 0]
        commander.publish_velocity(rc_command)
        time.sleep(0.01)
        '''
        rc_command = [20.0, 0.0, 0.0, 0.0]
        for _ in range(50):
            
            commander.publish_velocity(rc_command)
            time.sleep(0.1)
        '''
        rc_command = [0.0, 0.0, 0.0, 0.5]
        for _ in range(20):
            commander.publish_velocity(rc_command)
            time.sleep(0.1)
        
        rc_command = [0.0, 0.0, 0.0, 0.0]
        commander.publish_velocity(rc_command)
        time.sleep(0.2)
        '''
        '''
        rc_command = [20.0, 0.0, 0.0, 0.0]
        for _ in range(45):
            commander.publish_velocity(rc_command)
            time.sleep(0.1)
        '''
        rc_command = [0, 0, 0, 0]
        commander.publish_velocity(rc_command)
        time.sleep(1.0)

        commander.send_command("land")
        rclpy.spin_once(commander, timeout_sec=1.0)

    finally:
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
