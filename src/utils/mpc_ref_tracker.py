import numpy as np
import casadi as ca
from acados_template import AcadosOcp
import rclpy
from rclpy.node import Node

class MPCRefTracker(Node):
    def __init__(self) -> None:
        self.ocp = AcadosOcp()
        self.setup_dynamics()
        self.setup_constraints()
        self.setup_costs()
        self.setup_solver_options()
    
    def setup_dynamics(self):
        x = ca