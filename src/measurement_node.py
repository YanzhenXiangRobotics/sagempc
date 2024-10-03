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

import matplotlib.pyplot as plt


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

        self.s = MLSocket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

        self.dist_angle_sampling_num = 1000
        self.dist_buckets = [[] for _ in range(self.dist_angle_sampling_num)]

        self.begin = time.time()
        self.counter = 0

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
        world_2_base_link = self.tf_buffer.lookup_transform(
            "world", "base_link", time=rclpy.time.Time()
        )
        trans = world_2_base_link.transform.translation
        orient = world_2_base_link.transform.rotation
        orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
        orient_euler = np.array(euler_from_quaternion(orient_quat))
        pose_3D = np.array([trans.x, trans.y, orient_euler[-1]])
        pose_3D[-1] = self._angle_helper(pose_3D[-1])

        return pose_3D

    def convert_to_sim_time(self, clock):
        return clock.sec + 1e-9 * clock.nanosec

    def clock_listener_callback(self, msg):
        sim_time = self.convert_to_sim_time(msg.clock)
        try:
            self.pose_3D = self.get_pose_3D()
            # if self.min_dist != -1.0:
            #     data_to_send = np.concatenate(
            #         (
            #             self.pose_3D,
            #             np.array([self.min_dist_angle]),
            #             np.array([self.min_dist]),
            #             np.array([sim_time]),
            #         )
            #     )

            # print(f"To send {data_to_send}")

            # conn, _ = self.s.accept()
            # conn.sendall(data_to_send)
            # print(f"Sent {data_to_send}")
            # print(data_to_send)
            # conn.close()
        except Exception as e:
            print(e)

    def min_dist_listener_callback(self, msg):
        self.counter += 1
        try:
            ranges = np.array(msg.ranges)
            choice = np.round(
                np.linspace(1, len(ranges) - 1, num=self.dist_angle_sampling_num)
            ).astype(int)
            range_samples = ranges[choice]
            for i in range(self.dist_angle_sampling_num):
                self.dist_buckets[i].append(range_samples[i])
            dist_buckets_array = np.array(self.dist_buckets)
            for i in range(self.dist_angle_sampling_num):
                angle = (
                    -math.pi
                    + 2 * math.pi / self.dist_angle_sampling_num * i
                    + self.pose_3D[-1]
                )
                print(
                    f"Angle: {angle}, mean: {np.mean(dist_buckets_array[i, :])}, std: {np.std(dist_buckets_array[i, :])}"
                )
            print("\n")
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

    def plot_dist_mean_std(self):
        dist_buckets_array = np.array(self.dist_buckets)
        dist_mean = np.zeros((self.dist_angle_sampling_num,))
        dist_std = np.zeros((self.dist_angle_sampling_num,))
        for i in range(self.dist_angle_sampling_num):
            indices_nonzero = dist_buckets_array[i, :] > 0
            dist_mean[i] = np.mean(dist_buckets_array[i, :][indices_nonzero])
            dist_std[i] = np.std(dist_buckets_array[i, :][indices_nonzero])
        sorted_indices = np.argsort(dist_mean)
        dist_mean = dist_mean[sorted_indices]
        dist_std = dist_std[sorted_indices]
        inliers = dist_std < 0.05
        dist_mean_in, dist_std_in = dist_mean[inliers], dist_std[inliers]
        plt.figure()
        plt.plot(dist_mean, dist_std, marker="x")
        plt.show()
        # plt.figure()
        # plt.plot(dist_mean_in, dist_std_in)
        # plt.show()

if __name__ == "__main__":
    rclpy.init()
    node = MeasurementNode()
    while rclpy.ok() and node.counter < 100:
        rclpy.spin_once(node)

    node.plot_dist_mean_std()

    node.destroy_node()
    rclpy.shutdown()
