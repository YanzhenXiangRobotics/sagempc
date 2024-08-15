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

import numpy as np

HOST = "127.0.0.1"
PORT = 65432

import math
import time

class MeasurementNode(Node):
    def __init__(self):
        super().__init__("measurement_node")
        self.min_dist_subscriber = self.create_subscription(
            LaserScan, "/front_3d_lidar/scan", self.min_dist_listener_callback, 10
        )
        self.sim_time_subscriber = self.create_subscription(
            Float64, "/sim_time", self.sim_time_listener_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.min_dist = 6.0

        self.s = MLSocket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

        self.begin = time.time()

    def get_pose_3D(self):
        od_2_bl = self.tf_buffer.lookup_transform(
            "odom", "base_link", time=rclpy.time.Time()
        )
        trans = od_2_bl.transform.translation
        orient = od_2_bl.transform.rotation
        orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
        orient_euler = np.array(euler_from_quaternion(orient_quat))
        pose_3D = np.array([-trans.x, -trans.y, orient_euler[-1]])
        pose_3D += np.array([-20.0, -20.0, math.pi])

        return pose_3D

    def sim_time_listener_callback(self, msg):
        sim_time = msg.data
        try:
            pose_3D = self.get_pose_3D()
            data_to_send = np.concatenate(
                (pose_3D, np.array([self.min_dist]), np.array([sim_time]))
            )

            print(f"To send {data_to_send}")

            conn, _ = self.s.accept()
            conn.sendall(data_to_send)
            print(f"Sent {data_to_send}")
            # print(data_to_send)
            conn.close()
        except Exception as e:
            print(e)

    def min_dist_listener_callback(self, msg):
        try:
            ranges = np.array(msg.ranges)
            ranges = ranges[np.nonzero(ranges)]
            self.min_dist = np.min(ranges)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    rclpy.init()
    measure_distance_node = MeasurementNode()
    rclpy.spin(measure_distance_node)

    measure_distance_node.destroy_node()
    measure_distance_node.shutdown()
