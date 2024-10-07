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
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
import time
from src.utils.mpc_ref_tracker_node import MPCRefTracker
import math


def angle_helper(angle):
    if angle >= math.pi:
        return angle_helper(angle - 2 * math.pi)
    elif angle < -math.pi:
        return angle_helper(angle + 2 * math.pi)
    else:
        return angle


def get_current_pose(tf_buffer):
    pose_base_link = tf_buffer.lookup_transform(
        "world", "base_link", time=rclpy.time.Time()
    )
    trans = pose_base_link.transform.translation
    orient = pose_base_link.transform.rotation
    orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
    orient_euler = np.array(euler_from_quaternion(orient_quat))
    pose_3D = np.array([trans.x, trans.y, orient_euler[-1]])
    pose_3D[-1] = angle_helper(pose_3D[-1])

    return pose_3D


def estimate_velocity(curr_pose, last_pose, curr_time, last_time):
    dt = curr_time - last_time
    lin_vel = np.linalg.norm(curr_pose[:2] - last_pose[:2]) / 2
    ang_vel = 

class MainNode(Node):
    def __init__(self):
        super().__init__("main_node")
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock", self.pose_listener_callback, 10
        )
        

rclpy.init()
se_mpc = SEMPC(params, env, visu)

se_mpc.sempc_main()
print("avg time", np.mean(visu.iteration_time))
visu.save_data()

# X_test = np.zeros((30, 2))
# X_test[:, 0] = np.linspace(-20.0, -35.0, X_test.shape[0])
# X_test[:, 1] = np.linspace(-16.0, -31.0, X_test.shape[0])
# se_mpc.apply_control(X_test)
