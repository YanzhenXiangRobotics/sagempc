import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import math
import socket
from mlsocket import MLSocket
from src.world import World
from src.obstacle import Rectangle, Circle

HOST = "127.0.0.1"
PORT = 65432


class FakeSimulationNode(Node):
    def __init__(self):
        super().__init__("fake_simulation_node")
        self.subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.listener_callback, 10
        )
        self.timer = self.create_timer(1 / 10, self.on_timer)
        self.pose = np.array([-8.3, 4.0, math.pi])
        self.u = np.zeros(
            2,
        )
        self.t = 0
        self.dt = 1e-3
        self.world = World(
            bbox=[-14.45293, -16.74178, 9.54707, 22.0763], resolution=0.3
        )
        # self.world.add_obstacle(
        #     Rectangle(lower_left=[-2.0, -8.0], upper_right=[2.0, 8.0], resolution=self.world.resolution)
        # )
        self.world.add_obstacle(Circle(center=[0.0, 2.5], radius=2.0, resolution=self.world.resolution))

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
        self.t += self.dt
        self.dynamics()
        min_dist, _ = self.world.min_dist_to_obsc(self.pose[:2])
        data_to_send = np.concatenate(
            (self.pose, np.array([min_dist]), np.array([self.t]))
        )
        print(f"To send {data_to_send}")
        conn, _ = self.s.accept()
        conn.sendall(data_to_send)
        conn.close()
        print(f"Sent {data_to_send}")


if __name__ == "__main__":
    rclpy.init()
    sim = FakeSimulationNode()
    sim.world.plot(show=True)
    sim.setup_socket()
    rclpy.spin(sim)
