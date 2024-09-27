import socket
from mlsocket import MLSocket

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from tf_transformations import (
    euler_from_quaternion,
    quaternion_multiply,
)

from rosgraph_msgs.msg import Clock

import numpy as np

HOST = "127.0.0.1"
PORT = 65432

import math
import time

import os

dir_here = os.path.abspath(os.path.dirname(__file__))
import yaml

with open(
    os.path.join(
        dir_here,
        "..",
        "experiments",
        "nova_carter_isaac_sim",
        "env_0",
        "params_env.yaml",
    )
) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
with open(
    os.path.join(
        dir_here,
        "..",
        "params",
        "params_nova_carter_isaac_sim.yaml",
    )
) as file:
    params_0 = yaml.load(file, Loader=yaml.FullLoader)


class MeasurementNode(Node):
    def __init__(self):
        super().__init__("measurement_node")
        self.min_dist_subscriber = self.create_subscription(
            LaserScan, "/front_3d_lidar/scan", self.min_dist_listener_callback, 10
        )
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock", self.clock_listener_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.min_dist = -1.0
        self.last_sim_time = -1.0
        self.last_pose_3D = None

        self.s = MLSocket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

        self.begin = time.time()

    def _angle_helper(self, angle):
        if angle >= math.pi:
            return self._angle_helper(angle - 2 * math.pi)
        elif angle < -math.pi:
            return self._angle_helper(angle + 2 * math.pi)
        else:
            return angle

    # def get_pose_3D(self):
    #     # map_2_chassis_imu = self.tf_buffer.lookup_transform(
    #     #     "map", "chassis_imu", time=rclpy.time.Time()
    #     # )
    #     odom_2_chassis_imu = self.tf_buffer.lookup_transform(
    #         "odom", "chassis_imu", time=rclpy.time.Time()
    #     )
    #     trans = odom_2_chassis_imu.transform.translation
    #     trans = map_2_chassis_imu.transform.translation
    #     orient = map_2_chassis_imu.transform.rotation
    #     orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
    #     orient_euler = np.array(euler_from_quaternion(orient_quat))
    #     pose_3D = np.array([trans.x, trans.y, orient_euler[-1]])
    #     # start_pose = np.append(
    #     #     np.array(params["start_loc"]), params_0["env"]["start_angle"]
    #     # )
    #     # self.pose_3D += np.array(start_pose)
    #     pose_3D[-1] = self._angle_helper(pose_3D[-1])

    #     return pose_3D

    def get_pose_3D(self):
        world_2_chassis_imu = self.tf_buffer.lookup_transform(
            "world", "base_link", time=rclpy.time.Time()
        )
        trans = world_2_chassis_imu.transform.translation
        orient = world_2_chassis_imu.transform.rotation
        orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
        orient_euler = np.array(euler_from_quaternion(orient_quat))
        pose_3D = np.array([trans.x, trans.y, orient_euler[-1]])
        # start_pose = np.append(
        #     np.array(params["start_loc"]), params_0["env"]["start_angle"]
        # )
        # pose_3D += np.array(start_pose)
        pose_3D[-1] = self._angle_helper(pose_3D[-1])

        return pose_3D

    def convert_to_sim_time(self, clock):
        return clock.sec + 1e-9 * clock.nanosec

    def compute_velocity(self, dt, pose, last_pose):
        dtheta = pose[-1] - last_pose[-1]
        dtheta = dtheta - 2 * math.pi * round(dtheta / (2 * math.pi))
        dpos = np.linalg.norm(pose[:2] - last_pose[:2])
        darc = dpos if dtheta == 0.0 else dtheta * dpos / (2 * np.sin(dtheta / 2))
        v, omega = darc / dt, dtheta / dt
        return v, omega

    def clock_listener_callback(self, msg):
        sim_time = self.convert_to_sim_time(msg.clock)
        # print("Sim time: ", sim_time)
        try:
            if self.min_dist != -1.0:
                conn, _ = self.s.accept()
                self.pose_3D = self.get_pose_3D()
                if (self.last_pose_3D is not None) and (self.last_t != -1):
                    self.v, self.omega = self.compute_velocity(
                        sim_time - self.last_t, self.pose_3D, self.last_pose_3D
                    )
                    print(
                        f"Pose: {self.pose_3D} \n Last pose: {self.last_pose_3D} \n t: {sim_time} \n last_t: {self.last_t}"
                    )
                else:
                    self.v, self.omega = 0.0, 0.0
                self.last_pose_3D = self.pose_3D
                self.last_t = sim_time
                data_to_send = np.concatenate(
                    (
                        self.pose_3D,
                        np.array([self.v]),
                        np.array([self.omega]),
                        np.array([self.min_dist_angle]),
                        np.array([self.min_dist]),
                        np.array([sim_time]),
                    )
                )
                print(f"To send {data_to_send}")
                conn.sendall(data_to_send)
                conn.close()
        except Exception as e:
            print(e)

    def min_dist_listener_callback(self, msg):
        print("min_dist_listener_callback")
        try:
            ranges = np.array(msg.ranges)
            # choice = np.round(np.linspace(1, len(ranges)-1, num=36)).astype(int)
            # print(ranges[choice], "\n\n")
            ranges[ranges <= 0.0] += 1e3
            min_dist_idx = np.argmin(ranges)
            self.min_dist = ranges[min_dist_idx]
            self.min_dist_angle = (
                -math.pi + msg.angle_increment * min_dist_idx + self.pose_3D[-1]
                # -math.pi + msg.angle_increment * min_dist_idx
            )
            self.min_dist_angle = self._angle_helper(self.min_dist_angle)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    rclpy.init()
    measure_distance_node = MeasurementNode()
    rclpy.spin(measure_distance_node)

    measure_distance_node.destroy_node()
    measure_distance_node.shutdown()
