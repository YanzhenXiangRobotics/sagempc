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


class MeasurementNode(Node):
    def __init__(self):
        super().__init__("measure_distance_node")
        self.min_dist_subscriber = self.create_subscription(
            LaserScan, "/front_3d_lidar/scan", self.min_dist_listener_callback, 10
        )
        self.sim_time_subscriber = self.create_subscription(
            Float64, "/sim_time", self.sim_time_listener_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.s = MLSocket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

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

    def sim_time_listener_callback(self, msg):
        self.sim_time = msg.data

    def min_dist_listener_callback(self, msg):
        try:
            ranges = np.array(msg.ranges)
            ranges = ranges[np.nonzero(ranges)]
            min_dist = np.min(ranges)

            pose_3D = self.get_pose_3D()
            data_to_send = np.concatenate(
                (pose_3D, np.array([min_dist]), np.array([self.sim_time]))
            )

            print(f"To send {data_to_send}")

            conn, _ = self.s.accept()
            conn.sendall(data_to_send)
            print(f"Sent {data_to_send}")
            conn.close()
            

        except Exception as e:
            print(e)


if __name__ == "__main__":
    rclpy.init()
    measure_distance_node = MeasurementNode()
    rclpy.spin(measure_distance_node)

    measure_distance_node.destroy_node()
    measure_distance_node.shutdown()
