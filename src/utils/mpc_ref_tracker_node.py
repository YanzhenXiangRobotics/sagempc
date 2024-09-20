import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import rclpy
from rclpy.node import Node
from tf_transformations import (
    euler_from_quaternion,
)
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

import os
import yaml
import math
import time

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
        self.ocp.model.name = "nova_carter_discrete_Lc"
        self.H = params["optimizer"]["Hm"]
        self.ref_path = [params_additional["start_loc"] + [params["env"]["start_angle"]]]
        self.setup_dynamics()
        self.setup_constraints()
        self.setup_cost()
        self.setup_solver_options()

    def setup_dynamics(self):
        self.x_dim, self.u_dim = 2, 2
        x = ca.SX.sym("x", self.x_dim + 1)
        u = ca.SX.sym("u", self.u_dim)
        self.ocp.model.x, self.ocp.model.u = x, u

        v, omega, theta = u[0], u[1], x[2]
        self.dt = params["optimizer"]["dt"]
        K0 = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        K1 = ca.vertcat(
            v * ca.cos(theta + 0.5 * omega * self.dt),
            v * ca.sin(theta + 0.5 * omega * self.dt),
            omega,
        )
        K2 = ca.SX(K1)
        K3 = ca.vertcat(
            v * ca.cos(theta + omega * self.dt),
            v * ca.sin(theta + omega * self.dt),
            omega,
        )

        self.ocp.model.disc_dyn_expr = x + (self.dt / 6) * (K0 + 2 * K1 + 2 * K2 + K3)

    def setup_constraints(self):
        self.ocp.constraints.lbx = np.array(params["optimizer"]["x_min"])
        self.ocp.constraints.ubx = np.array(params["optimizer"]["x_max"])
        self.ocp.constraints.idxbx = np.arange(self.x_dim)
        
        self.ocp.constraints.x0 = np.array(self.ref_path[0])

        self.ocp.constraints.lbx_e = self.ocp.constraints.lbx.copy()
        self.ocp.constraints.ubx_e = self.ocp.constraints.ubx.copy()
        self.ocp.constraints.idxbx_e = self.ocp.constraints.idxbx.copy()

        self.ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
        self.ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
        self.ocp.constraints.idxbu = np.arange(self.u_dim)

    def setup_cost(self):
        x_ref = ca.SX.sym("x_ref", self.x_dim + 1)
        self.ocp.model.p = x_ref
        self.ocp.parameter_values = np.zeros((self.ocp.model.p.shape[0],))

        Q = np.eye(self.x_dim + 1)
        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.cost.cost_type_e = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = (
            (self.ocp.model.x - x_ref).T
            @ Q
            @ (self.ocp.model.x - x_ref)
        )
        self.ocp.model.cost_expr_ext_cost_e = (
            (self.ocp.model.x - x_ref).T
            @ Q
            @ (self.ocp.model.x - x_ref)
        )

    def setup_solver_options(self):
        self.ocp.dims.N = self.H
        self.ocp.solver_options.tf = params["optimizer"]["Tf"]
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.levenberg_marquardt = 1.0e-1
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.nlp_solver_ext_qp_res = 1
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"

        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp_sempc.json")

    def solver_set_ref_path(self):
        for k in range(self.H + 1):
            if k < len(self.ref_path):
                self.ocp_solver.set(k, "p", np.array(self.ref_path[k]))
            else:
                self.ocp_solver.set(k, "p", np.array(self.ref_path[-1]))
        
        
    def get_solution(self):
        X = np.zeros((self.H + 1, self.x_dim + 1))
        U = np.zeros((self.H, self.u_dim))
        for k in range(self.H):
            X[k, :] = self.ocp_solver.get(k, "x")
            U[k, :] = self.ocp_solver.get(k, "u")
        X[-1, :] = self.ocp_solver.get(self.H, "x")
        print(f"X: {X}, U: {U}")
        return X, U
    
    def solve_for_x0(self, x0):
        self.ocp_solver.options_set("rti_phase", 1)
        self.solver_set_ref_path()
        status = self.ocp_solver.solve()
        
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)
        
        self.ocp_solver.options_set("rti_phase", 2)
        status = self.ocp_solver.solve()
        
        X, U = self.get_solution()
        
        return U[0, :]

    def update_ref_path(self, new_ref_path):
        self.ref_path += new_ref_path
        print(f"Ref path: {self.ref_path}")


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
    
    def get_pose_3D(self):
        map_2_chassis_imu = self.tf_buffer.lookup_transform(
            "map", "chassis_imu", time=rclpy.time.Time()
        )
        odom_2_chassis_imu = self.tf_buffer.lookup_transform(
            "odom", "chassis_imu", time=rclpy.time.Time()
        )
        trans = odom_2_chassis_imu.transform.translation
        orient = map_2_chassis_imu.transform.rotation
        orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
        orient_euler = np.array(euler_from_quaternion(orient_quat))
        self.pose_3D = np.array([-trans.x, -trans.y, orient_euler[-1]])
        start_pose = np.append(
            np.array(params_additional["start_loc"]), params["env"]["start_angle"]
        )
        self.pose_3D += np.array(start_pose)
        self.pose_3D[-1] = self._angle_helper(self.pose_3D[-1] - math.pi)
        self.init_pose_obtained = True
        
    def compute_sim_time(self, clock):
        return clock.sec + 1e-9 * clock.nanosec

    def clock_listener_callback(self, msg):
        if self.sim_time == -1.0:
            self.last_sim_time = self.compute_sim_time(msg.clock)
        self.sim_time = self.compute_sim_time(msg.clock)
        try:
            self.get_pose_3D()
            # if self.min_dist != -1.0:
            #     data_to_send = np.concatenate(
            #         (
            #             self.pose_3D,
            #             np.array([self.min_dist_angle]),
            #             np.array([self.min_dist]),
            #             np.array([self.sim_time]),
            #         )
            #     )

            #     print(f"To send {data_to_send}")

            #     conn, _ = self.s.accept()
            #     conn.sendall(data_to_send)
            #     print(f"Sent {data_to_send}")
            #     # print(data_to_send)
            #     conn.close()
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
        u = self.ctrl.solve_for_x0(self.pose_3D)
        time_after = time.time()
        cmd_vel = Twist()
        cmd_vel.linear.x = u[0]
        cmd_vel.angular.z = u[1]
        self.publisher.publish(cmd_vel)

    def ref_path_listener_callback(self, msg):
        new_ref_path = np.array(msg.data).reshape(self.ctrl.H + 1, -1).tolist()
        self.ctrl.update_ref_path(new_ref_path)


if __name__ == "__main__":
    rclpy.init()
    mpc_ref_tracker_node = MPCRefTrackerNode()
    rclpy.spin(mpc_ref_tracker_node)

    mpc_ref_tracker_node.destroy_node()
    rclpy.shutdown()
