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
    true_objective_func=opt.true_density,
    opt_goal=opt.opt_goal,
    optimal_feasible_boundary=opt.optimal_feasible_boundary,
    params=params,
    path=save_path + str(traj_iter),
)

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import time
from src.utils.inner_control_node import InnerControl
import math
from src.utils.inner_control_node import (
    get_current_pose,
    compute_velocity_fwk_nova_carter,
)


class PlannerNode(Node):
    def __init__(self):
        super().__init__("main_node")
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock_planner", self.clock_listener_callback, 10
        )
        self.LiDAR_subscriber = self.create_subscription(
            LaserScan, "/front_3d_lidar/scan", self.LiDAR_listener_callback, 10
        )
        self.velocity_subscriber = self.create_subscription(
            JointState, "/joint_states", self.velocity_listener_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ref_path_publisher = self.create_publisher(
            Float32MultiArray, "/ref_path", 10
        )
        self.sempc_initialized = False
        self.LiDAR_meas_obtained = False
        self.pose_curr = np.append(
            np.array(params["env"]["start_loc"]), params["env"]["start_angle"]
        )
        self.curr_loc_min_dists = []
        self.noise_scaling_factor = 5e-3

    def update_pose_curr(self):
        pose_curr = get_current_pose(self.tf_buffer)
        if pose_curr is not None:
            self.pose_curr = pose_curr

    def clock_listener_callback(self, msg):
        before = time.time()
        if self.LiDAR_meas_obtained:
            if not self.sempc_initialized:
                self.sempc = SEMPC(
                    params,
                    env,
                    visu,
                    query_state_obs_noise=np.concatenate(
                        (
                            self.pose_curr[: params["common"]["dim"]],
                            np.array(
                                [
                                    self.min_dist,
                                    self.noise_scaling_factor * self.min_dist,
                                ]
                            ),
                        )
                    ),
                )
                self.sempc_initialized = True
            else:
                loc_curr = self.pose_curr[: self.sempc.x_dim]
                state_curr = np.concatenate((self.pose_curr, self.velocity))

                self.curr_loc_min_dists.append(
                    np.concatenate(
                        (
                            self.pose_curr[: params["common"]["dim"]],
                            np.array(
                                [
                                    self.min_dist,
                                    self.noise_scaling_factor * self.min_dist,
                                ]
                            ),
                        )
                    )
                )

                self.obs_arr = np.concatenate(
                    (
                        self.obs_arr,
                        np.array(self.curr_loc_min_dists),
                    ),
                    axis=0,
                )

                if self.sempc.running_condition_true_go(loc_curr):
                    self.sempc.update_Cx_gp(self.obs_arr)
                    self.sempc.set_next_goal(loc_curr)
                    X_ol, _ = self.sempc.one_step_planner(state_curr)

                    ref_path_cmd = Float32MultiArray()
                    ref_path_cmd.data = (
                        X_ol[:, : self.sempc.pose_dim].flatten().tolist()
                    )
                    self.ref_path_publisher.publish(ref_path_cmd)
        print("Time planner: ", time.time() - before)

    def velocity_listener_callback(self, msg):
        omega_wl, omega_wr = msg.velocity[1], msg.velocity[2]
        self.velocity = compute_velocity_fwk_nova_carter(omega_wl, omega_wr)

    def LiDAR_listener_callback(self, msg):
        ranges = np.array(msg.ranges)

        self.update_pose_curr()
        subsample_num = 200
        subsample_indices = np.round(
            np.linspace(1, len(ranges) - 1, num=subsample_num)
        ).astype(int)

        ranges_subsampled = ranges[subsample_indices]
        angles_subsampled = (
            self.pose_curr[-1] + msg.angle_increment * subsample_indices - math.pi
        )

        valid_indices = ranges_subsampled > 0.0
        ranges_subsampled = ranges_subsampled[valid_indices]
        angles_subsampled = angles_subsampled[valid_indices]

        self.obs_arr = np.zeros(
            (ranges_subsampled.shape[0], params["common"]["dim"] + 2)
        )
        self.obs_arr[:, 0] = self.pose_curr[0] + ranges_subsampled * np.cos(
            angles_subsampled
        )
        self.obs_arr[:, 1] = self.pose_curr[1] + ranges_subsampled * np.sin(
            angles_subsampled
        )
        self.obs_arr[:, -1] = self.noise_scaling_factor * ranges_subsampled

        ranges_copy = ranges.copy()
        ranges_copy[ranges_copy <= 0.0] += 1e3
        min_dist_idx = np.argmin(ranges_copy)
        self.min_dist = ranges[min_dist_idx]

        self.LiDAR_meas_obtained = True


if __name__ == "__main__":
    if params["experiment"]["use_isaac_sim"] == 1:
        rclpy.init()
        planner = PlannerNode()
        rclpy.spin(planner)

        planner.destroy_node()
        rclpy.shutdown()
    else:
        sempc = SEMPC(params, env, visu)

        sempc.sempc_main()
        print("avg time", np.mean(visu.iteration_time))
        visu.save_data()
