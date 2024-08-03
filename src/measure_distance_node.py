import socket

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import numpy as np

HOST = "127.0.0.1"
PORT = 65432


class MeasureDistanceNode(Node):
    def __init__(self):
        super.__init__("measure_distance_node")
        self.subscriber = self.create_subscription(
            LaserScan, "/front_3d_lidar/scan", self.listener_callback, 10
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            self.conn, addr = s.accept()
            with self.conn:
                print(f"Connected by {addr}")

    def listener_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = ranges[np.nonzero(ranges)]
        min_dist = np.min(ranges)
        with self.conn:
            self.conn.sendall(min_dist)