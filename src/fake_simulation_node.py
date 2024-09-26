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


class FakeSimulationNode(Node):
    def __init__(self):
        super().__init__("fake_simulation_node")
        self.subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.listener_callback, 10
        )
        self.timer = self.create_timer(1 / 100, self.on_timer)
        self.pose = np.append(
            np.array(params["start_loc"]), params_0["env"]["start_angle"]
        )
        self.u = np.zeros(
            2,
        )
        self.t = -1.0
        # self.dt = 0.1
        start = params_0["env"]["start"]
        step_size = params_0["visu"]["step_size"]
        end = [
            start[0] + params_0["env"]["shape"]["x"] * step_size,
            start[1] + params_0["env"]["shape"]["y"] * step_size,
        ]
        self.world = World(bbox=start + end, resolution=step_size)
        self.world.add_obstacle(
            Rectangle(
                lower_left=[-18.5, -22.0],
                upper_right=[-11.0, -15.5],
                resolution=self.world.resolution,
            )
        )
        self.world.add_obstacle(
            Rectangle(
                lower_left=[-20.8, -14.0],
                upper_right=[-16.0, -13.0],
                resolution=self.world.resolution,
            )
        )
        # self.world.add_obstacle(Circle(center=[0.0, 2.5], radius=2.0, resolution=self.world.resolution))
        # self.world.add_obstacle(
        #     DiamondSquare(
        #         center=[-18.0, -18.0],
        #         radius=1.0 / math.sqrt(2),
        #         resolution=self.world.resolution,
        #     )
        # )
        # self.world.add_obstacle(
        #     Rectangle(
        #         lower_left=[-21.0, -16.5],
        #         upper_right=[-20.8, -16.0],
        #         resolution=self.world.resolution,
        #     )
        # )
        self.begin = time.time()

    def setup_socket(self):
        self.s = MLSocket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((HOST, PORT))
        self.s.listen(5)

    def listener_callback(self, msg):
        self.u = np.array([msg.linear.x, msg.angular.z])
        # print(self.u)

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
            # if self.dt > 2.0:
            #     print(self.u)
            self.dynamics()
            min_dist, min_dist_angle = self.world.min_dist_to_obsc(self.pose[:2])
            # min_dist += np.random.uniform(-0.05, 0.05)
            # min_dist_angle += np.random.uniform(-0.1, 0.1)
            data_to_send = np.concatenate(
                (
                    self.pose,
                    np.array([min_dist_angle]),
                    np.array([min_dist]),
                    np.array([self.t]),
                )
            )
            # print(f"To send {data_to_send}")
            conn, _ = self.s.accept()
            conn.sendall(data_to_send)
            conn.close()
            self.t = time.time() - self.begin
            # print(f"Sent {data_to_send}")


if __name__ == "__main__":
    rclpy.init()
    sim = FakeSimulationNode()
    sim.world.plot(show=False)
    sim.setup_socket()
    rclpy.spin(sim)
