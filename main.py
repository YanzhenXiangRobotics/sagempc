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

from src.environement import ContiWorld
from src.ground_truth import GroundTruth
from src.SEMPC_nova_carter import SEMPCNovaCarter 
from src.utils.helper import (TrainAndUpdateConstraint, TrainAndUpdateDensity,
                              get_frame_writer, oracle)
from src.utils.initializer import get_players_initialized
from src.utils.plotting import plot_1D, plot_2D
from src.visu import Visu

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [12, 6]

workspace = "sagempc"

parser = argparse.ArgumentParser(description='A foo that bars')
parser.add_argument('-param', default="params_nova_carter_isaac_sim")  # params
parser.add_argument('-env', type=int, default=0)
parser.add_argument('-i', type=int, default=8)  # initialized at origin
args = parser.parse_args()

import os
dir_project = os.path.abspath(os.path.dirname(__file__))

# 1) Load the config file
with open(dir_project + "/params/" + args.param + ".yaml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
params["env"]["i"] = args.i
params["env"]["name"] = args.env
print(params)

import rclpy
rclpy.init()
se_mpc = SEMPCNovaCarter(params)
se_mpc.sempc_main()