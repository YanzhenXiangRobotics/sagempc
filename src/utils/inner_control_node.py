import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.model import export_nova_carter_discrete_rk4_fixedtime

import rclpy
from rclpy.node import Node
from tf_transformations import (
    euler_from_quaternion,
)
from std_msgs.msg import Float64, Float32MultiArray, Int32
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, JointState

import os
import yaml
import math

import matplotlib.pyplot as plt

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


class InnerControl:
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
        self.pose_dim = self.x_dim + 1
        self.H = params["innerloop"]["H"]
        self.N = params["innerloop"]["N"]
        self.ref_path = []
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

        self.ocp.constraints.x0 = np.array(
            params_additional["start_loc"]
            + [params["env"]["start_angle"]]
            + np.zeros(self.x_dim).tolist()
        )

        self.ocp.constraints.lbx_e = self.ocp.constraints.lbx.copy()
        self.ocp.constraints.lbx_e[-self.x_dim :] = np.zeros(self.x_dim)
        self.ocp.constraints.ubx_e = self.ocp.constraints.ubx.copy()
        self.ocp.constraints.ubx_e[-self.x_dim :] = np.zeros(self.x_dim)
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx.copy()

        self.ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
        self.ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
        self.ocp.constraints.idxbu = np.arange(self.u_dim)

    def setup_cost(self):
        x_ref = ca.SX.sym("x_ref", self.pose_dim)
        self.ocp.model.p = x_ref
        self.ocp.parameter_values = np.zeros((self.ocp.model.p.shape[0],))

        Q = np.diag([1e4, 1e4, 1.0])
        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.cost.cost_type_e = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = (
            (self.ocp.model.x[: self.pose_dim] - x_ref).T
            @ Q
            @ (self.ocp.model.x[: self.pose_dim] - x_ref)
        )
        self.ocp.model.cost_expr_ext_cost_e = (
            (self.ocp.model.x[: self.pose_dim] - x_ref).T
            @ Q
            @ (self.ocp.model.x[: self.pose_dim] - x_ref)
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

    def get_solution(self):
        X = np.zeros((self.H + 1, self.pose_dim + self.x_dim))
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
            for k in range(self.H + 1):
                self.ocp_solver.set(k, "p", np.array(self.ref_path[0]))
            status = self.ocp_solver.solve()

            self.ocp_solver.set(0, "lbx", x0)
            self.ocp_solver.set(0, "ubx", x0)

            self.ocp_solver.options_set("rti_phase", 2)

            status = self.ocp_solver.solve()
            if self.debug:
                print(
                    f"SQP iter: {i}, Cost: {self.ocp_solver.get_cost()}, Res: {self.ocp_solver.get_residuals()}"
                )

        X, U = self.get_solution()

        return X, U

    def set_ref_path(self, ref_path):
        self.ref_path = ref_path
        if self.debug:
            print(f"Ref path: {np.array(self.ref_path)}")


# from main import get_current_pose, compute_velocity_fwk_nova_carter
def clip_angle(angle):
    if angle >= math.pi:
        return clip_angle(angle - 2 * math.pi)
    elif angle < -math.pi:
        return clip_angle(angle + 2 * math.pi)
    else:
        return angle


def get_current_pose(tf_buffer):
    try:
        pose_base_link = tf_buffer.lookup_transform(
            "world", "base_link", time=rclpy.time.Time()
        )
        trans = pose_base_link.transform.translation
        orient = pose_base_link.transform.rotation
        orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
        orient_euler = np.array(euler_from_quaternion(orient_quat))
        pose_3D = np.array([trans.x, trans.y, orient_euler[-1]])
        pose_3D[-1] = clip_angle(pose_3D[-1])
        return pose_3D
    except Exception as e:
        print(e)
        return None


def compute_velocity_fwk_nova_carter(omega_wl, omega_wr):
    l = 2 * 0.2066
    r = 0.14

    v_wl, v_wr = r * omega_wl, r * omega_wr
    v = 0.5 * (v_wl + v_wr)
    omega = (v_wr - v_wl) / l

    return np.array([v, omega])


import os
import shutil


class InnerControlPlotter(Node):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.X_ol, self.X_cl = [], []
        self.local_radius = 0.3
        dir_project = os.path.join(os.path.dirname(__file__), "..", "..")
        self.dir_saveplots = os.path.join(dir_project, "inner_loop_isaac_sim")
        if os.path.exists(self.dir_saveplots):
            shutil.rmtree(self.dir_saveplots)
        os.makedirs(self.dir_saveplots)
        self.plots_list = []

    def add_to_openloop(self, X):
        self.X_ol.append(X)

    def add_to_closeloop(self, X):
        self.X_cl.append(X)

    def plot_openloop(self):
        X_ol = np.array(self.X_ol)
        if len(X_ol) != 0:
            (plot,) = self.ax.plot(X_ol[:, 0], X_ol[:, 1], marker="x", color="black")
            self.plots_list.append(plot)

    def plot_closeloop(self):
        X_cl = np.array(self.X_cl)
        if len(X_cl) != 0:
            (plot,) = self.ax.plot(X_cl[:, 0], X_cl[:, 1], marker="o", color="lime")
            self.plots_list.append(plot)

    def save_fig(self, iter):
        self.ax.set_xlim(
            [
                self.X_cl[-1][0] - self.local_radius,
                self.X_cl[-1][0] + self.local_radius,
            ]
        )
        self.ax.set_ylim(
            [
                self.X_cl[-1][1] - self.local_radius,
                self.X_cl[-1][1] + self.local_radius,
            ]
        )
        outer_iter = math.floor(iter / params["optimizer"]["H"])
        inner_iter = iter % params["optimizer"]["H"]
        self.fig.savefig(
            os.path.join(self.dir_saveplots, f"{outer_iter}_{inner_iter}.png")
        )
        len_plots_list = len(self.plots_list)
        for _ in range(len_plots_list):
            self.plots_list.pop(0).remove()


class InnerControlNode(Node):
    def __init__(self):
        super().__init__("mpc_ref_tracker_node")
        self.ctrl = InnerControl()
        self.clock_subscriber = self.create_subscription(
            Clock, "/clock_controller", self.clock_listener_callback, 10
        )
        self.ref_path_subscriber = self.create_subscription(
            Float32MultiArray, "/ref_path", self.ref_path_listener_callback, 10
        )
        # self.min_dist_subscriber = self.create_subscription(
        #     LaserScan, "/front_3d_lidar/scan", self.min_dist_listener_callback, 10
        # )
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.velocity_subscriber = self.create_subscription(
            JointState, "/joint_states", self.velocity_listener_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.iter = -1

        self.init_pose_obtained = False
        self.ref_path_init = False

        self.debug_plot = params["innerloop"]["debug_plot"]
        if self.debug_plot:
            self.plotter = InnerControlPlotter()

    def clock_listener_callback(self, msg):
        try:
            pose = get_current_pose(self.tf_buffer)
            # self.X_cl[self.iter, :] = pose[: self.ctrl.x_dim]
            if not self.ref_path_init:
                self.ctrl.ref_path.append(pose[: self.ctrl.pose_dim])
                self.ref_path_init = True

            if (self.iter % self.ctrl.N == 0) and (len(self.ctrl.ref_path) > 1):
                self.ctrl.ref_path.pop(0)

            X, _ = self.ctrl.solve_for_x0(np.concatenate((pose, self.velocity)))
            cmd_vel = Twist()
            if self.iter != -1:
                cmd_vel.linear.x = X[1, self.ctrl.pose_dim]
                cmd_vel.angular.z = X[1, self.ctrl.pose_dim + 1]
            self.cmd_vel_publisher.publish(cmd_vel)
            
            if self.iter != -1:
                self.iter += 1
                if (self.debug_plot) and (self.iter % self.ctrl.N == 0):
                    # print(f"Iter: {self.iter}")
                    self.plotter.add_to_closeloop(pose[: self.ctrl.x_dim].tolist())
                    self.plotter.plot_closeloop()
                    self.plotter.plot_openloop()
                    self.plotter.save_fig(math.floor(self.iter / self.ctrl.N ))

        except Exception as e:
            print(e)

    def velocity_listener_callback(self, msg):
        omega_wl, omega_wr = msg.velocity[1], msg.velocity[2]
        self.velocity = compute_velocity_fwk_nova_carter(omega_wl, omega_wr)
        # print(f"Velocity: {self.velocity}")

    def ref_path_listener_callback(self, msg):
        print("Ref path updated")
        if self.iter == -1:
            self.iter = 0
        ref_path = np.array(msg.data).reshape(params["optimizer"]["H"] + 1, -1).tolist()
        self.ctrl.set_ref_path(ref_path)
        for ref_path_item in ref_path:
            self.plotter.add_to_openloop(ref_path_item)

    def plot(self, ref_path):
        ref_path_appended = np.zeros_like(self.X_cl)
        ref_path_appended[: self.X_cl.shape[0], :] = ref_path.copy()
        ref_path_appended[ref_path.shape[0] : self.X_cl.shape[0], :] = ref_path[-1, :]

        _, ax = plt.subplots()
        ax.plot(
            ref_path_appended[:, 0], ref_path_appended[:, 1], marker="o", color="lime"
        )
        ax.plot(self.X_cl[:, 0], self.X_cl[:, 1], marker="x", color="black")
        print(self.X_cl - ref_path_appended)
        plt.show()


if __name__ == "__main__":
    rclpy.init()
    controller = InnerControlNode()

    if not params["innerloop"]["sanity_check"]:
        rclpy.spin(controller)
    else:
        ref_path = np.array(
            [
                [-20.0, -16.0, 3.142, 0.0, 0.0, 0.0],
                [-19.997, -16.0, 3.136, -0.123, -0.236, 0.049],
                [-19.991, -16.0, 3.125, -0.211, -0.411, 0.084],
                [-19.982, -16.0, 3.108, -0.298, -0.586, 0.119],
                [-19.97, -16.001, 3.084, -0.386, -0.761, 0.154],
                [-19.955, -16.002, 3.054, -0.473, -0.936, 0.189],
                [-19.938, -16.004, 3.02, -0.5, -1.0, 0.224],
                [-19.923, -16.006, 2.985, -0.419, -1.0, 0.259],
                [-19.91, -16.008, 2.95, -0.332, -1.0, 0.294],
                [-19.9, -16.01, 2.915, -0.244, -1.0, 0.329],
                [-19.889, -16.013, 2.803, 0.059, -1.0, 0.451],
                [-19.892, -16.012, 2.768, 0.147, -1.0, 0.486],
                [-19.898, -16.009, 2.733, 0.234, -1.0, 0.521],
                [-19.907, -16.006, 2.698, 0.322, -1.0, 0.556],
                [-19.919, -16.0, 2.663, 0.409, -1.0, 0.591],
                [-19.933, -15.993, 2.628, 0.497, -1.0, 0.626],
                [-19.948, -15.984, 2.593, 0.5, -1.0, 0.661],
                [-19.974, -15.969, 2.534, 0.5, -1.0, 0.72],
                [-19.988, -15.959, 2.499, 0.5, -1.0, 0.755],
                [-20.001, -15.949, 2.466, 0.438, -0.875, 0.79],
                [-20.012, -15.94, 2.438, 0.35, -0.7, 0.825],
                [-20.021, -15.933, 2.417, 0.263, -0.525, 0.86],
                [-20.027, -15.927, 2.402, 0.175, -0.35, 0.895],
                [-20.03, -15.924, 2.392, 0.088, -0.175, 0.93],
                [-2.004e01, -1.592e01, 2.469e00, 0.000e00, 0.000e00, 0.000e00],
                [-2.004e01, -1.592e01, 2.464e00, -1.126e-01, -2.251e-01, 4.503e-02],
                [-2.004e01, -1.592e01, 2.449e00, -1.558e-01, -4.532e-01, 1.719e-01],
                [-2.003e01, -1.592e01, 2.429e00, -2.065e-01, -6.331e-01, 2.069e-01],
                [-2.003e01, -1.593e01, 2.403e00, -1.761e-01, -8.130e-01, 2.419e-01],
                [-2.002e01, -1.593e01, 2.372e00, -8.672e-02, -9.378e-01, 2.769e-01],
                [-2.002e01, -1.593e01, 2.337e00, 2.789e-03, -1.000e00, 3.119e-01],
                [-2.002e01, -1.593e01, 2.301e00, 9.071e-02, -1.000e00, 3.487e-01],
                [-2.002e01, -1.593e01, 2.265e00, 8.474e-02, -1.000e00, 3.837e-01],
                [-2.003e01, -1.592e01, 2.229e00, 1.568e-01, -1.000e00, 4.187e-01],
                [-2.003e01, -1.592e01, 2.193e00, 1.497e-01, -1.000e00, 4.537e-01],
                [-2.003e01, -1.591e01, 2.157e00, 2.396e-01, -1.000e00, 4.887e-01],
                [-2.004e01, -1.591e01, 2.121e00, 3.280e-01, -1.000e00, 5.237e-01],
                [-2.004e01, -1.589e01, 2.085e00, 3.909e-01, -1.000e00, 5.587e-01],
                [-2.005e01, -1.588e01, 2.049e00, 4.527e-01, -1.000e00, 5.937e-01],
                [-2.006e01, -1.587e01, 2.013e00, 5.000e-01, -1.000e00, 6.287e-01],
                [-2.007e01, -1.585e01, 1.977e00, 5.000e-01, -1.000e00, 6.637e-01],
                [-2.008e01, -1.584e01, 1.941e00, 5.000e-01, -1.000e00, 6.987e-01],
                [-2.010e01, -1.581e01, 1.884e00, 5.000e-01, -1.000e00, 7.550e-01],
                [-2.011e01, -1.580e01, 1.850e00, 4.467e-01, -8.996e-01, 7.900e-01],
                [-2.011e01, -1.579e01, 1.821e00, 3.572e-01, -7.196e-01, 8.250e-01],
                [-2.012e01, -1.578e01, 1.798e00, 2.678e-01, -5.397e-01, 8.600e-01],
                [-2.012e01, -1.577e01, 1.782e00, 1.785e-01, -3.598e-01, 8.950e-01],
                [-2.012e01, -1.577e01, 1.772e00, 8.920e-02, -1.799e-01, 9.300e-01],
            ]
        )
        max_iter = 48 * controller.ctrl.N
        # controller.X_cl = np.zeros((max_iter, controller.ctrl.x_dim))
        controller.ctrl.set_ref_path(ref_path[:, :-3].tolist())
        controller.iter = 0

        for ref_path_item in ref_path[:, :-3].tolist():
            controller.plotter.add_to_openloop(ref_path_item)
        while controller.iter < max_iter:
            rclpy.spin_once(controller)
        # controller.plot(ref_path[:, :2])

    controller.destroy_node()
    rclpy.shutdown()
