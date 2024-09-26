# The environment for this file is a ~/work/rl
import argparse
import errno
import os
import warnings
from datetime import datetime

import casadi as ca
import gpytorch
import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np

import os, sys

dir_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_here)

from src.environment import ContiWorld
from src.ground_truth import GroundTruth
from src.SEMPC import SEMPC
from src.utils.helper import (
    TrainAndUpdateConstraint,
    get_frame_writer,
    oracle,
)
from src.utils.initializer import get_players_initialized
from src.utils.plotting import plot_1D, plot_2D
from src.visu import Visu
from src.utils.helper import train_and_update_constraint

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [12, 6]

workspace = "sagempc"

parser = argparse.ArgumentParser(description="A foo that bars")
parser.add_argument("-param", default="params_nova_carter_isaac_sim")  # params
# parser.add_argument('-param', default="params_cluttered_car")
parser.add_argument("-env", type=int, default=0)
parser.add_argument("-i", type=int, default=8)  # initialized at origin
args = parser.parse_args()

import sys, os

dir_project = os.path.abspath(os.path.dirname(__file__))
import shutil

if os.path.exists(os.path.join(dir_project, "sqp_sols")):
    shutil.rmtree(os.path.join(dir_project, "sqp_sols"))
inner_loop_plot_path = os.path.join(dir_project, "inner_loop")
if os.path.exists(inner_loop_plot_path):
    shutil.rmtree(inner_loop_plot_path)
os.makedirs(inner_loop_plot_path)

# 1) Load the config file
with open(dir_project + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = args.i
params["env"]["name"] = args.env
# if params["agent"]["dynamics"] == "nova_carter":
#     params["optimizer"]["dt"] = 0.035
# params["optimizer"]["dt"] = 0.9 * params["optimizer"]["Tf"] / params["optimizer"]["H"]
print(params)

# 2) Set the path and copy params from file
exp_name = params["experiment"]["name"]
env_load_path = (
    dir_project
    + "/experiments/"
    + params["experiment"]["folder"]
    + "/env_"
    + str(args.env)
    + "/"
)

save_path = env_load_path + "/" + args.param + "/"

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# set start and the goal location
with open(env_load_path + "/params_env.yaml") as file:
    env_st_goal_pos = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["start_loc"] = env_st_goal_pos["start_loc"]
if len(params["env"]["start_loc"]) > 2:
    params["env"]["start_loc"] = params["env"]["start_loc"][:2]
params["env"]["goal_loc"] = env_st_goal_pos["goal_loc"]

# 3) Setup the environment. This class defines different environments eg: wall, forest, or a sample from GP.
env = ContiWorld(
    env_params=params["env"],
    common_params=params["common"],
    visu_params=params["visu"],
    env_dir=env_load_path,
    params=params,
)

opt = GroundTruth(env, params)

if params["env"]["compute_true_Lipschitz"]:
    print(env.get_true_lipschitz())
    exit()

print(args)
if args.i != -1:
    traj_iter = args.i

if not os.path.exists(save_path + str(traj_iter)):
    os.makedirs(save_path + str(traj_iter))

visu = Visu(
    grid_V=env.VisuGrid,
    safe_boundary=env.get_safe_init()["Cx_X"],
    true_constraint_function=opt.true_constraint_function,
    opt_goal=opt.opt_goal,
    optimal_feasible_boundary=opt.optimal_feasible_boundary,
    params=params,
    path=save_path + str(traj_iter),
)

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import time
from src.utils.mpc_ref_tracker_node import MPCRefTracker


class MainNode(Node):
    def __init__(self):
        super().__init__("main_node")
        self.timer = self.create_timer(1 / 100, self.on_timer)
        self.mpc_tracking_timer = self.create_timer(1 / 20, self.mpc_tracking)
        # self.clock_subscriber = self.create_subscription(
        #     Clock, "/clock", self.clock_listener_callback, 10
        # )
        # if params["experiment"]["use_fake_sim"]:
        self.pose_subscriber = self.create_subscription(
            Float32MultiArray, "/pose", self.pose_listener_callback, 10
        )
        self.velocity_subscriber = self.create_subscription(
            Float32MultiArray, "/vel", self.velocity_listener_callback, 10
        )
        self.min_dist_subscriber = self.create_subscription(
            Float32MultiArray, "/min_dist", self.min_dist_listener_callback, 10
        )
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sempc = SEMPC(params, env, visu, self.cmd_vel_publisher)
        self.w = 100
        self.running_condition_true = True
        self.ref_tracker = MPCRefTracker()
        self.x_dim = self.sempc.x_dim
        self.state_dim = self.sempc.state_dim
        self.phase = 0
        self.sempc_initialized = False
        self.min_dist_obtained = False
        self.pose_obtained = False
        self.velocity_obtained = False

    def on_timer(self):
        if self.sempc_initialized and (self.phase == 0):
            if self.running_condition_true:
                train_and_update_constraint(
                    self.query_pts,
                    self.query_meas,
                    self.sempc.pl_idx,
                    self.sempc.players,
                    self.sempc.params,
                )

                if self.w < self.sempc.params["common"]["epsilon"]:
                    self.sempc.players[self.sempc.pl_idx].feasible = False
                else:
                    ckp = time.time()
                    self.w = self.sempc.set_next_goal()
                    print(f"Time for finding next goal: {time.time() - ckp}")

                if self.sempc.params["algo"]["objective"] == "GO":
                    self.running_condition_true = (
                        not np.linalg.norm(
                            self.sempc.visu.utility_minimizer
                            - self.sempc.players[self.sempc.pl_idx].current_location
                        )
                        < self.sempc.params["visu"]["step_size"]
                    )
                elif self.sempc.params["algo"]["objective"] == "SE":
                    self.running_condition_true = self.sempc.players[
                        self.sempc.pl_idx
                    ].feasible
                else:
                    raise NameError("Objective is not clear")
                print(
                    "Number of samples",
                    self.sempc.players[self.sempc.pl_idx].Cx_X_train.shape,
                )
                X, _ = self.sempc.one_step_planner(x_curr=self.pose)
                self.ref_tracker.set_ref_path(X.tolist())
            else:
                print("avg time", np.mean(visu.iteration_time))
                visu.save_data()
                exit()
            self.phase = 1

    def mpc_tracking(self):
        if (self.phase == 1) and self.pose_obtained and self.velocity_obtained:
            x0 = np.concatenate((self.pose, np.array([self.v]), np.array([self.omega])))
            X_inner, U_inner = self.ref_tracker.solve_for_x0(x0)
            msg_cmd_vel = Twist()
            msg_cmd_vel.linear.x = X_inner[1, self.state_dim]
            msg_cmd_vel.angular.z = X_inner[1, self.state_dim + 1]
            begin = time.time()
            while time.time() - begin < U_inner[1, -1]:
                self.cmd_vel_publisher.publish(msg_cmd_vel)
                print("Publishing: ", msg_cmd_vel.linear.x, msg_cmd_vel.angular.z)
            if len(self.ref_tracker.ref_path) == 0:
                self.phase = 0

    # def clock_listener_callback(self, msg):
    #     t_curr = msg.clock.sec + 1e-9 * msg.clock.nanosec
    #     self.sempc.update_sim_time(t_curr)

    def pose_listener_callback(self, msg):
        self.pose_obtained = True
        print("Get pose: ", msg.data)
        self.pose = np.array([msg.data[0], msg.data[1], msg.data[2]])
        if (not self.sempc_initialized) and (self.min_dist_obtained):
            self.sempc.sempc_initialization(self.query_pts, self.query_meas)
            self.sempc.players[self.sempc.pl_idx].feasible = True
            self.sempc_initialized = True
        if self.sempc_initialized:
            self.sempc.players[self.sempc.pl_idx].update_current_state(self.pose)
    
    def velocity_listener_callback(self, msg):
        self.velocity_obtained = True
        self.v, self.omega = msg.data[0], msg.data[1]
        print("Get v: ", self.v, " omega: ", self.omega)

    def min_dist_listener_callback(self, msg):
        min_dist, min_dist_angle = msg.data[0], msg.data[1]
        print("Get min dist: ", min_dist, " min dist angle: ", min_dist_angle)
        self.min_dist_obtained = True
        query_pts_x_start = self.pose[0]
        query_pts_x_end = self.pose[0] + min_dist * np.cos(min_dist_angle)
        num_pts = params["experiment"]["batch_size"]
        query_pts_x = np.linspace(
            query_pts_x_start,
            query_pts_x_end,
            num_pts,
        )
        query_pts_y = np.linspace(
            self.pose[1],
            self.pose[1] + (query_pts_x[-1] - query_pts_x[0]) * np.tan(min_dist_angle),
            len(query_pts_x),
        )
        self.query_pts = np.vstack((query_pts_x, query_pts_y)).T
        self.query_meas = np.linspace(
            min_dist,
            min_dist - (query_pts_x[-1] - query_pts_x[0]) / np.cos(min_dist_angle),
            len(query_pts_x),
        )


if __name__ == "__main__":
    rclpy.init()
    main_node = MainNode()
    rclpy.spin(main_node)

    main_node.destroy_node()
    main_node.shutdown()

# se_mpc.sempc_main()
# print("avg time", np.mean(visu.iteration_time))
# visu.save_data()

# X_test = np.zeros((30, 2))
# X_test[:, 0] = np.linspace(-20.0, -35.0, X_test.shape[0])
# X_test[:, 1] = np.linspace(-16.0, -31.0, X_test.shape[0])
# se_mpc.apply_control(X_test)
