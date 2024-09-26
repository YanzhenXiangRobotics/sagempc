import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32MultiArray
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
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_listener_callback, 10
        )
        self.simulation_timer = self.create_timer(1 / 100, self.simulation_on_timer)
        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self.pose_publisher = self.create_publisher(Float32MultiArray, "/pose", 10)
        self.min_dist_publisher = self.create_publisher(
            Float32MultiArray, "/min_dist", 10
        )
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

    def cmd_vel_listener_callback(self, msg):
        self.u = np.array([msg.linear.x, msg.angular.z])
        # print(self.u)

    def dynamics(self):
        print("Running dynamics: ", self.u, self.dt)
        self.pose = np.array(
            [
                self.pose[0] + self.u[0] * np.cos(self.pose[2]) * self.dt,
                self.pose[1] + self.u[0] * np.sin(self.pose[2]) * self.dt,
                self.pose[2] + self.u[1] * self.dt,
            ]
        )

    def simulation_on_timer(self):
        # begin = time.time()
        # while time.time() - begin < 10.0:
        #     pass
        last_t = self.t
        self.t = time.time() - self.begin
        if last_t != -1.0:
            self.dt = self.t - last_t
            self.dynamics()
            min_dist, min_dist_angle = self.world.min_dist_to_obsc(self.pose[:2])
            self.publish_clock()
            self.publish_pose()
            self.publish_min_dist(min_dist, min_dist_angle)

    def publish_clock(self):
        # print("Clock: ", self.t)
        msg_clock = Clock()
        msg_clock.clock.sec = math.floor(self.t)
        msg_clock.clock.nanosec = int((self.t - msg_clock.clock.sec) * 1e9)
        self.clock_publisher.publish(msg_clock)

    def publish_pose(self):
        msg_pose = Float32MultiArray()
        msg_pose.data = self.pose.tolist()
        self.pose_publisher.publish(msg_pose)

    def publish_min_dist(self, min_dist, min_dist_angle):
        msg_min_dist = Float32MultiArray()
        msg_min_dist.data = [min_dist, min_dist_angle]
        self.min_dist_publisher.publish(msg_min_dist)


if __name__ == "__main__":
    rclpy.init()
    sim = FakeSimulationNode()
    sim.world.plot(show=False)
    rclpy.spin(sim)
