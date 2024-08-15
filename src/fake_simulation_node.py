import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import math
import socket
from mlsocket import MLSocket
import sys, os

dir_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_here)
from utils.world import World
from utils.obstacle import Rectangle, DiamondSquare

HOST = "127.0.0.1"
PORT = 65432

import time

class FakeSimulationNode(Node):
    def __init__(self):
        super().__init__("fake_simulation_node")
        self.subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.listener_callback, 10
        )
        self.timer = self.create_timer(1 / 100, self.on_timer)
        self.pose = np.array([-20.0, -20.0, math.pi])
        self.u = np.zeros(
            2,
        )
        self.t = -1.0
        # self.dt = 0.1
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
        # self.world.add_obstacle(Circle(center=[0.0, 2.5], radius=2.0, resolution=self.world.resolution))
        self.world.add_obstacle(
            DiamondSquare(
                center=[-18.0, -18.0],
                radius=1.0 / math.sqrt(2),
                resolution=self.world.resolution,
            )
        )
        self.world.add_obstacle(
            Rectangle(
                lower_left=[-21.8, -16.5],
                upper_right=[-21.3, -15.5],
                resolution=self.world.resolution,
            )
        )
        self.begin = time.time()

    def setup_socket(self):
        self.s = MLSocket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

    def listener_callback(self, msg):
        self.u = np.array([msg.linear.x, msg.angular.z])

    def dynamics(self):
        self.pose = np.array(
            [
                self.pose[0] + self.u[0] * np.cos(self.pose[2]) * self.dt,
                self.pose[1] + self.u[0] * np.sin(self.pose[2]) * self.dt,
                self.pose[2] + self.u[1] * self.dt,
            ]
        )

    def on_timer(self):
        last_t = self.t
        self.t = time.time() - self.begin
        if last_t != -1.0:
            self.dt = self.t - last_t
            # print(self.dt)
            if self.dt > 2.0:
                print(self.u)
            self.dynamics()
            min_dist, _ = self.world.min_dist_to_obsc(self.pose[:2])
            data_to_send = np.concatenate(
                (self.pose, np.array([min_dist]), np.array([self.t]))
            )
            # print(f"To send {data_to_send}")
            conn, _ = self.s.accept()
            conn.sendall(data_to_send)
            conn.close()
            # print(f"Sent {data_to_send}")


if __name__ == "__main__":
    rclpy.init()
    sim = FakeSimulationNode()
    sim.world.plot(show=False)
    sim.setup_socket()
    rclpy.spin(sim)
