import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from src.utils.model import export_nova_carter_discrete_rk4_fixedtime

import rclpy
from rclpy.node import Node
from tf_transformations import (
    euler_from_quaternion,
)
from std_msgs.msg import Float64, Float64MultiArray, Int32
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

import os
import yaml
import math
import time

from mlsocket import MLSocket

HOST = "127.0.0.1"
PORT = 65432

dir_here = os.path.abspath(os.path.dirname(__file__))
with open(
    os.path.join(
        dir_here,
        "..",
        "..",
        "params",
        "params_nova_carter_isaac_sim.yaml",
    )
) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
with open(
    os.path.join(
        dir_here,
        "..",
        "..",
        "experiments",
        "nova_carter_isaac_sim",
        "env_0",
        "params_env.yaml",
    )
) as file:
    params_additional = yaml.load(file, Loader=yaml.FullLoader)


class MPCRefTracker:
    def __init__(self) -> None:
        self.ocp = AcadosOcp()
        self.ocp.model = AcadosModel()
        self.ocp.model.name = "nova_carter_discrete_Lc_inner_loop"

        self.init_params()
        self.setup_dynamics()
        self.setup_cost()
        self.setup_constraints()
        self.setup_solver_options()

    def init_params(self):
        self.x_dim, self.u_dim = 2, 2
        self.state_dim = self.x_dim + 1
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.ref_path = [
            params_additional["start_loc"] + [params["env"]["start_angle"]] + [0.0, 0.0]
        ]
        self.w_terminal = 1.0
        self.w_horizons = np.linspace(1.0, self.w_terminal, self.H + 1)
        self.lbx_middle = np.concatenate(
            (
                params["optimizer"]["x_min"][: self.x_dim],
                np.zeros(self.x_dim),
            )
        )
        self.ubx_middle = np.concatenate(
            (params["optimizer"]["x_max"][: self.x_dim], np.zeros(self.x_dim))
        )
        self.lbx_final = self.lbx_middle.copy()
        self.ubx_final = self.ubx_middle.copy()
        self.pos_scale_base = 1e-3
        self.debug = params["experiment"]["debug"]

    def setup_dynamics(self):
        self.ocp.model = export_nova_carter_discrete_rk4_fixedtime(
            params["optimizer"]["Tf"] / params["optimizer"]["H"]
        )
        self.ocp.model.name = "inner_loop_" + self.ocp.model.name

    def setup_constraints(self):
        self.ocp.constraints.lbx = np.array(params["optimizer"]["x_min"])
        self.ocp.constraints.ubx = np.array(params["optimizer"]["x_max"])
        self.ocp.constraints.idxbx = np.array([0, 1, 3, 4])

        self.ocp.constraints.x0 = np.array(self.ref_path[0])

        self.ocp.constraints.lbx_e = np.concatenate(
            (
                self.ocp.constraints.lbx.copy()[: self.x_dim],
                np.zeros(self.x_dim),
            )
        )
        self.ocp.constraints.ubx_e = np.concatenate(
            (
                self.ocp.constraints.ubx.copy()[: self.x_dim],
                np.zeros(self.x_dim),
            )
        )
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx.copy()

        self.ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
        self.ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
        self.ocp.constraints.idxbu = np.arange(self.u_dim)

    def setup_cost(self):
        x_ref = ca.SX.sym("x_ref", self.state_dim)
        self.ocp.model.p = x_ref
        self.ocp.parameter_values = np.zeros((self.ocp.model.p.shape[0],))

        Q = np.diag([1e4, 1e4, 1.0])
        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.cost.cost_type_e = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = (
            (self.ocp.model.x[: self.state_dim] - x_ref).T
            @ Q
            @ (self.ocp.model.x[: self.state_dim] - x_ref)
        )
        self.ocp.model.cost_expr_ext_cost_e = self.w_terminal * (
            (self.ocp.model.x[: self.state_dim] - x_ref).T
            @ Q
            @ (self.ocp.model.x[: self.state_dim] - x_ref)
        )

    def setup_solver_options(self):
        self.ocp.dims.N = self.H
        self.ocp.solver_options.tf = params["optimizer"]["Tf"]
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.levenberg_marquardt = 1.0e-2
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.nlp_solver_ext_qp_res = 1
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"

        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file="inner_loop_acados_ocp_sempc.json"
        )

    def solver_set_ref_path(self):
        for k in range(self.H + 1):
            if k < len(self.ref_path):
                self.ocp_solver.set(k, "p", np.array(self.ref_path[k]))
            else:
                w_pos = self.pos_scale_base
                (self.ocp_solver.set(k, "p", np.array(self.ref_path[-1])))

    def get_solution(self):
        X = np.zeros((self.H + 1, self.state_dim + self.x_dim))
        U = np.zeros((self.H, self.u_dim))
        for k in range(self.H):
            X[k, :] = self.ocp_solver.get(k, "x")
            U[k, :] = self.ocp_solver.get(k, "u")
        X[-1, :] = self.ocp_solver.get(self.H, "x")
        # print(f"X: {X}, U: {U}")
        return X, U

    def solve_for_x0(self, x0):
        for i in range(3):
            self.ocp_solver.options_set("rti_phase", 1)
            self.solver_set_ref_path()
            status = self.ocp_solver.solve()

            self.ocp_solver.set(0, "lbx", x0)
            self.ocp_solver.set(0, "ubx", x0)
            self.ocp_solver.set(self.Hm, "lbx", self.lbx_middle)
            self.ocp_solver.set(self.Hm, "ubx", self.ubx_middle)
            self.ocp_solver.set(self.H, "lbx", self.lbx_final)
            self.ocp_solver.set(self.H, "ubx", self.ubx_final)

            self.ocp_solver.options_set("rti_phase", 2)

            status = self.ocp_solver.solve()
            if self.debug:
                print(
                    f"SQP iter: {i}, Cost: {self.ocp_solver.get_cost()}, Res: {self.ocp_solver.get_residuals()}"
                )

        X, U = self.get_solution()
        self.ref_path.pop(0)

        return X, U

    def set_ref_path(self, ref_path):
        self.ref_path = ref_path
        if self.debug:
            print(f"Ref path: {np.array(self.ref_path)}")

from main import get_current_pose, estimate_velocity
class MPCRefTrackerNode(Node):
    def __init__(self):
        super().__init__("mpc_ref_tracker_node")
        self.ctrl = MPCRefTracker()
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock_controller", self.clock_listener_callback, 10
        )
        self.ref_path_subscriber = self.create_subscription(
            Float64MultiArray, "/ref_path", self.ref_path_listener_callback, 10
        )
        # self.min_dist_subscriber = self.create_subscription(
        #     LaserScan, "/front_3d_lidar/scan", self.min_dist_listener_callback, 10
        # )
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1 / 100, self.check_event)

        self.sim_time, self.last_sim_time = -1.0, -1.0

        self.init_pose_obtained = False

    def _angle_helper(self, angle):
        if angle >= math.pi:
            return self._angle_helper(angle - 2 * math.pi)
        elif angle < -math.pi:
            return self._angle_helper(angle + 2 * math.pi)
        else:
            return angle
    
    def clock_listener_callback(self, msg):
        curr_time = msg.sec + 1e-9 * msg.nano_sec
        pose = get_current_pose(self.tf_buffer)
        velocity = estimate_velocity(pose, self.last_pose, curr_time, self.last_time)
        self.last_pose = pose
        self.last_time = curr_time
       
        X, U = self.ctrl.solve_for_x0(np.concatenate((pose, velocity)))
        
        cmd_vel = Twist()
        cmd_vel.linear.x = X[1, self.ctrl.state_dim]
        cmd_vel.angular.z = X[1, self.ctrl.state_dim + 1]
        self.cmd_vel_publisher.publish(cmd_vel)

    def ref_path_listener_callback(self, msg):
        new_ref_path = np.array(msg.data).reshape(self.ctrl.H + 1, -1).tolist()
        self.ctrl.set_ref_path(new_ref_path)


if __name__ == "__main__":
    rclpy.init()
    mpc_ref_tracker_node = MPCRefTrackerNode()
    rclpy.spin(mpc_ref_tracker_node)

    mpc_ref_tracker_node.destroy_node()
    rclpy.shutdown()
