import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import math
import sys, os

dir_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_here)
from utils.world import World
from utils.obstacle import Rectangle
from std_msgs.msg import Float64

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from tf_transformations import (
    euler_from_quaternion,
    quaternion_multiply,
)


class FakeUseSimTimeNode(Node):
    def __init__(self):
        super().__init__("fake_simulation_node")
        self.twist_subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.twist_callback, 10
        )
        self.sim_time_subscriber = self.create_subscription(
            Float64, "/sim_time", self.sim_time_callback, 10
        )
        # self.pose = np.array([-20.0, -20.0, math.pi])
        self.u = np.zeros(
            2,
        )
        self.t = -1.0
        self.world = World(bbox=[-21.8, -21.8, 2.1, 2.1], resolution=0.2)
        self.world.add_obstacle(
            Rectangle(
                lower_left=[-14.0, -22.0],
                upper_right=[-11.0, -16.0],
                resolution=self.world.resolution,
            )
        )
        self.world.add_obstacle(
            Rectangle(
                lower_left=[-22.0, -14.0],
                upper_right=[-16.0, -9.0],
                resolution=self.world.resolution,
            )
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.init_pose_obtained = False
        # self.world.add_obstacle(Circle(center=[0.0, 2.5], radius=2.0, resolution=self.world.resolution))

    def get_pose_3D(self):
        map_2_odom = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time())
        odom_2_base_link = self.tf_buffer.lookup_transform(
            "odom", "base_link", time=rclpy.time.Time()
        )
        quat_map_2_odom = np.array(
            [
                map_2_odom.transform.rotation.x,
                map_2_odom.transform.rotation.y,
                map_2_odom.transform.rotation.z,
                map_2_odom.transform.rotation.w,
            ]
        )
        quat_odom_2_base_link = np.array(
            [
                odom_2_base_link.transform.rotation.x,
                odom_2_base_link.transform.rotation.y,
                odom_2_base_link.transform.rotation.z,
                odom_2_base_link.transform.rotation.w,
            ]
        )
        orient = quaternion_multiply(quat_map_2_odom, quat_odom_2_base_link)
        translation_2D = np.array(
            [
                map_2_odom.transform.translation.x
                - odom_2_base_link.transform.translation.x,
                map_2_odom.transform.translation.y
                - odom_2_base_link.transform.translation.y,
            ]
        )
        orient = np.array(euler_from_quaternion(orient))
        pose_3D = np.append(translation_2D, orient[2])

        return pose_3D
    
    def twist_callback(self, msg):
        self.u = np.array([msg.linear.x, msg.angular.z])
        if not self.init_pose_obtained:
            try:
                self.pose = self.get_pose_3D()
                self.init_pose_obtained = True
            except Exception as e:
                print(e)

    def sim_time_callback(self, msg):
        if self.init_pose_obtained:
            last_t = self.t
            self.t = msg.data
            if last_t != -1.0:
                self.dt = self.t - last_t
                self.dynamics()
                min_dist, _ = self.world.min_dist_to_obsc(self.pose[:2])
                data_to_send = np.concatenate(
                    (self.pose, np.array([min_dist]), np.array([self.t]))
                )
                print(data_to_send)
            

    def dynamics(self):
        self.pose = np.array(
            [
                self.pose[0] + self.u[0] * np.cos(self.pose[2]) * self.dt,
                self.pose[1] + self.u[0] * np.sin(self.pose[2]) * self.dt,
                self.pose[2] + self.u[1] * self.dt,
            ]
        )


if __name__ == "__main__":
    rclpy.init()
    sim = FakeUseSimTimeNode()
    sim.world.plot(show=False)
    rclpy.spin(sim)
