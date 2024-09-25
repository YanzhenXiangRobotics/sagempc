import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
from src.utils.model import export_nova_carter_discrete_Lc_rk4

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
            params_additional["start_loc"]
            + [params["env"]["start_angle"]]
            + [0.0, 0.0, 0.0]
        ]
        self.w_terminal = 10.0
        self.w_horizons = np.linspace(1.0, self.w_terminal, self.H + 1)
        self.lbx_middle = np.concatenate(
            (
                params["optimizer"]["x_min"][: self.x_dim],
                np.zeros(self.x_dim),
                np.array([0.0]),
            )
        )
        self.ubx_middle = np.concatenate(
            (
                params["optimizer"]["x_max"][: self.x_dim],
                np.zeros(self.x_dim),
                np.array([params["optimizer"]["Tf"]]),
            )
        )
        self.lbx_final = self.lbx_middle.copy()
        self.ubx_final = self.ubx_middle.copy()
        self.pos_scale_base = 1e-3

    def setup_dynamics(self):
        self.ocp.model = export_nova_carter_discrete_Lc_rk4()
        self.ocp.model.u = self.ocp.model.u[:-2]
        self.ocp.model.name = "inner_loop_" + self.ocp.model.name

    def setup_constraints(self):
        self.ocp.constraints.lbx = np.append(
            np.array(params["optimizer"]["x_min"]), 0.0
        )
        self.ocp.constraints.ubx = np.append(
            np.array(params["optimizer"]["x_max"]), params["optimizer"]["Tf"]
        )
        self.ocp.constraints.idxbx = np.array([0, 1, 3, 4, 5])

        self.ocp.constraints.x0 = np.array(self.ref_path[0])

        self.ocp.constraints.lbx_e = np.concatenate(
            (
                self.ocp.constraints.lbx.copy()[: self.x_dim],
                np.zeros(self.x_dim),
                np.array([params["optimizer"]["Tf"]]),
            )
        )
        self.ocp.constraints.ubx_e = np.concatenate(
            (
                self.ocp.constraints.ubx.copy()[: self.x_dim],
                np.zeros(self.x_dim),
                np.array([params["optimizer"]["Tf"]]),
            )
        )
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx.copy()

        self.ocp.constraints.lbu = np.append(
            np.array(params["optimizer"]["u_min"]), params["optimizer"]["dt"]
        )
        self.ocp.constraints.ubu = np.append(
            np.array(params["optimizer"]["u_max"]), params["optimizer"]["Tf"]
        )
        self.ocp.constraints.idxbu = np.arange(self.u_dim + 1)

    def setup_cost(self):
        x_ref = ca.SX.sym("x_ref", self.state_dim + self.x_dim + 1)
        w_speed = ca.SX.sym("w", 1)
        w_pos = ca.SX.sym("w_pos", 1)
        w_horizon = ca.SX.sym("w_horizon", 1)
        self.ocp.model.p = ca.vertcat(x_ref, w_speed, w_pos, w_horizon)
        self.ocp.parameter_values = np.zeros((self.ocp.model.p.shape[0],))

        Q = np.diag(
            [1e4 * w_pos, 1e4 * w_pos, 1.0, 4.0 * w_speed, 10.0 * w_speed, 10.0]
        )
        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.cost.cost_type_e = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = w_horizon * (
            (self.ocp.model.x - x_ref).T @ Q @ (self.ocp.model.x - x_ref)
        )
        # self.ocp.model.cost_expr_ext_cost_e = 5.0 * (
        #     (self.ocp.model.x - x_ref).T @ Q @ (self.ocp.model.x - x_ref)
        # )
        self.ocp.model.cost_expr_ext_cost_e = self.w_terminal * (
            (self.ocp.model.x - x_ref).T @ Q @ (self.ocp.model.x - x_ref)
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
        self.T_final = self.ref_path[-1][-1]
        for k in range(self.H + 1):
            if k < len(self.ref_path):
                w_pos = (
                    self.pos_scale_base
                    + 100.0
                    * np.linalg.norm(
                        np.array(self.ref_path[k][: self.x_dim])
                        - np.array(self.ref_path[k - 1][: self.x_dim])
                    )
                    if k > 0
                    else self.pos_scale_base
                )
                self.ocp_solver.set(
                    k,
                    "p",
                    np.concatenate(
                        (
                            np.array(self.ref_path[k]),
                            np.array([1.0, w_pos, self.w_horizons[k]]),
                        )
                    ),
                )
            else:
                w_pos = self.pos_scale_base
                self.T_final += params["optimizer"]["dt"]
                self.ocp_solver.set(
                    k,
                    "p",
                    np.concatenate(
                        (
                            np.array(self.ref_path[-1])[:-1],
                            np.array([self.T_final, 0.0, w_pos, self.w_horizons[k]]),
                        )
                    ),
                )
            print(f"Stage: {k}, Pos scale: {w_pos}")

    def get_solution(self):
        X = np.zeros((self.H + 1, self.state_dim + self.x_dim + 1))
        U = np.zeros((self.H, self.u_dim + 1))
        for k in range(self.H):
            X[k, :] = self.ocp_solver.get(k, "x")
            U[k, :] = self.ocp_solver.get(k, "u")
        X[-1, :] = self.ocp_solver.get(self.H, "x")
        # print(f"X: {X}, U: {U}")
        return X, U

    def ref_path_zero_init_time(self):
        T0 = self.ref_path[0][-1]
        for k in range(len(self.ref_path)):
            self.ref_path[k][-1] -= T0

    def solve_for_x0(self, x0):
        self.ref_path_zero_init_time()
        for i in range(5):
            self.ocp_solver.options_set("rti_phase", 1)
            self.solver_set_ref_path()
            status = self.ocp_solver.solve()

            self.ocp_solver.set(0, "lbx", np.append(x0, 0.0))
            self.ocp_solver.set(0, "ubx", np.append(x0, 0.0))
            self.ocp_solver.set(self.Hm, "lbx", self.lbx_middle)
            self.ocp_solver.set(self.Hm, "ubx", self.ubx_middle)
            self.ocp_solver.set(self.H, "lbx", self.lbx_final)
            self.ocp_solver.set(self.H, "ubx", self.ubx_final)

            self.ocp_solver.options_set("rti_phase", 2)

            status = self.ocp_solver.solve()
            print(
                f"SQP iter: {i}, Cost: {self.ocp_solver.get_cost()}, Res: {self.ocp_solver.get_residuals()}"
            )

        X, U = self.get_solution()
        self.ref_path.pop(0)

        return X, U

    def update_ref_path(self, new_ref_path):
        self.ref_path += new_ref_path
        print(f"Ref path: {np.array(self.ref_path)}")

    def set_ref_path(self, ref_path):
        self.ref_path = ref_path
        print(f"Ref path: {np.array(self.ref_path)}")


class MPCRefTrackerNode(Node):
    def __init__(self):
        super().__init__("mpc_ref_tracker_node")
        self.ctrl = MPCRefTracker()
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock", self.clock_listener_callback, 10
        )
        self.ref_path_subscriber = self.create_subscription(
            Float64MultiArray, "/ref_path", self.ref_path_listener_callback, 10
        )
        # self.min_dist_subscriber = self.create_subscription(
        #     LaserScan, "/front_3d_lidar/scan", self.min_dist_listener_callback, 10
        # )
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.complete_publisher = self.create_publisher(Int32, "/complete", 10)
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

    # def min_dist_listener_callback(self, msg):
    #     try:
    #         ranges = np.array(msg.ranges)
    #         # choice = np.round(np.linspace(1, len(ranges)-1, num=36)).astype(int)
    #         # print(ranges[choice], "\n\n")
    #         ranges[ranges<=0.0] += 1e3
    #         min_dist_idx = np.argmin(ranges)
    #         self.min_dist = ranges[min_dist_idx]
    #         self.min_dist_angle = (
    #             -math.pi + msg.angle_increment * min_dist_idx + self.pose_3D[-1]
    #             # -math.pi + msg.angle_increment * min_dist_idx
    #         )
    #         self.min_dist_angle = self._angle_helper(self.min_dist_angle)

    #     except Exception as e:
    #         print(e)

    def get_curr_pose_clock(self):
        try:
            s = MLSocket()
            s.connect((HOST, PORT))
            data = s.recv(1024)
            s.close()

            self.pose_3D = data[: self.ctrl.state_dim]
            self.sim_time = data[-1]

        except Exception as e:
            print(e)

    def check_event(self):
        if (
            self.init_pose_obtained
            and (self.sim_time != -1.0)
            and (self.last_sim_time != -1.0)
        ):
            if self.sim_time - self.last_sim_time >= self.ctrl.dt:
                print("Solving...")
                self.control_callback()
                if len(self.ctrl.ref_path) > 1:
                    self.ctrl.ref_path.pop(0)
                self.last_sim_time = self.sim_time
            else:
                print("Waiting...")
                zero_vel_cmd = Twist()
                # self.publisher.publish(zero_vel_cmd)

    def control_callback(self):
        time_before = time.time()
        self.get_curr_pose_clock()
        u = self.ctrl.solve_for_x0(self.pose_3D)
        time_after = time.time()
        cmd_vel = Twist()
        cmd_vel.linear.x = u[0]
        cmd_vel.angular.z = u[1]
        self.publisher.publish(cmd_vel)

    def ref_path_listener_callback(self, msg):
        new_ref_path = np.array(msg.data).reshape(self.ctrl.H + 1, -1).tolist()
        self.ctrl.update_ref_path(new_ref_path)
        complete_msg = Int32()
        if np.array(self.ctrl.ref_path).shape[0] == 1:
            complete_msg.data = 1
        else:
            complete_msg.data = 0
        self.complete_publisher.publish(complete_msg)


if __name__ == "__main__":
    rclpy.init()
    mpc_ref_tracker_node = MPCRefTrackerNode()
    rclpy.spin(mpc_ref_tracker_node)

    mpc_ref_tracker_node.destroy_node()
    rclpy.shutdown()
