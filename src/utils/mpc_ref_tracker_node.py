import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import rclpy
from rclpy.node import Node
from tf_transformations import (
    euler_from_quaternion,
)
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import os
import yaml
import math

dir_here = os.path.abspath(os.path.dirname(__file__))
with open(
    os.path.join(
        dir_here,
        "..",
        "params",
        "params_nova_carter_isaac_sim.yaml",
    )
) as file:
    params_0 = yaml.load(file, Loader=yaml.FullLoader)
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


class MPCRefTracker():
    def __init__(self) -> None:
        self.ocp = AcadosOcp()
        self.ocp.model = AcadosModel()
        self.ocp.model.name = "nova_carter_discrete_Lc"
        self.setup_dynamics()
        self.setup_constraints()
        self.setup_cost()
        self.setup_solver_options()
        self.H = params["optimizer"]["H"]
        self.ref_path = []

    def setup_dynamics(self):
        x = ca.SX.sym("x", 4)
        u = ca.SX.sym("u", 3)
        self.ocp.model.x, self.ocp.model.u = x, u

        v, omega, theta = u[0], u[1], x[2]
        self.dt = params_0["optimizer"]["dt"]
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
        self.ocp.constraints.lbx = np.array(params_0["optimizer"]["x_min"])
        self.ubx = np.array(params_0["optimizer"]["x_max"])
        self.x_dim = self.lbx.shape[0]
        self.ocp.constraints.ldxbx = np.arange(self.x_dim)

        self.lbu = np.array(params_0["optimizer"]["u_min"])
        self.ubu = np.array(params_0["optimizer"]["u_max"])
        self.u_dim = self.lbu.shape[0]
        self.idxbu = np.arange(self.u_dim)

    def setup_cost(self):
        x_ref = ca.SX.sym("x_ref", self.x_dim)
        self.ocp.model.p = x_ref
        self.ocp.parameter_values = np.zeros((self.ocp.model.p.shape[0],))
        
        Q = np.eye(self.x_dim)
        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.cost.cost_type_e = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = (self.ocp.model.x - x_ref).T @ Q(
            self.ocp.model.x - x_ref
        )
        self.ocp.model.cost_expr_ext_cost_e = (self.ocp.model.x - x_ref).T @ Q(
            self.ocp.model.x - x_ref
        )

    def setup_solver_options(self):
        self.ocp.dims.N = params["optimizer"]["H"]
        self.ocp.solver_options.tf = params["optimizer"]["Tf"]
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.levenberg_marquardt = 1.0e-1
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.nlp_solver_ext_qp_res = 1
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        
        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file="acados_ocp_sempc.json"
        )
        
    def solve_for_x0(self, x0):
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)
        self.ocp.constraints.x0 = x0
        for k in range(self.H):
            self.ocp_solver.set(k, "p", self.ref_path[k])
        u = self.ocp_solver.solve_for_x0(x0)
        return u
    
    def update_ref_path(self, new_ref_path):
        self.ref_path += new_ref_path
    
class MPCRefTrackerNode(Node):
    def __init__(self) -> None:
        super().init("mpc_ref_tracker_node")
        self.ctrl = MPCRefTracker()
        self.sim_time_subscriber = self.create_subscription(
            Float64, "/sim_time", self.sim_time_listener_callback, 10
        )
        self.ref_path_subscriber = self.create_subscription(
            Float64, "/ref_path", self.ref_path_listener_callback, 10
        )
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1 / 100, self.check_event)
        
        self.sim_time, self.last_sim_time = -1.0, -1.0
        
        while True:
            try:
                self.get_pose_3D()
                break
            except Exception as e:
                print(e)
        
    def _angle_helper(self, angle):
        if angle >= math.pi:
            return self._angle_helper(angle - 2 * math.pi)
        elif angle < -math.pi:
            return self._angle_helper(angle + 2 * math.pi)
        else:
            return angle
        
    def get_pose_3D(self):
        odom_2_chassis_imu = self.tf_buffer.lookup_transform(
            "odom", "chassis_imu", time=rclpy.time.Time()
        )
        trans = odom_2_chassis_imu.transform.translation
        orient = odom_2_chassis_imu.transform.rotation
        orient_quat = np.array([orient.x, orient.y, orient.z, orient.w])
        orient_euler = np.array(euler_from_quaternion(orient_quat))
        self.pose_3D = np.array([-trans.x, -trans.y, orient_euler[-1]])
        start_pose = np.append(
            np.array(params["start_loc"]), params_0["env"]["start_angle"]
        )
        self.pose_3D += np.array(start_pose)
        self.pose_3D[-1] = self._angle_helper(self.pose_3D[-1])
        
    def sim_time_listener_callback(self, msg):
        if self.sim_time == -1.0:
            self.last_sim_time = msg.data
        self.sim_time = msg.data
            
    def check_event(self):
        if (self.sim_time != -1.0) and (self.last_sim_time != -1.0):
            if self.sim_time - self.last_sim_time >= self.ctrl.dt:
                self.control_callback()
                self.ctrl.ref_path.pop(0)
                self.last_sim_time = self.sim_time
                
    def control_callbck(self):
        u = self.ctrl.solve_for_x0(self.pose_3D)
        cmd_vel = Twist()
        cmd_vel.linear.x = u[0]
        cmd_vel.angular.z = u[1]
        self.publisher.publish(cmd_vel)
                
    def ref_path_listener_callback(self, msg):
        self.ctrl.update_ref_path(msg.data)
            
    