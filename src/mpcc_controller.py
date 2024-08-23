from acados_template import AcadosOcp, AcadosOcpSolver
import casadi as ca
import numpy as np

class MPCC_controller():
    def __init__(self) -> None:
        self.ocp = AcadosOcp()
        self.ocp.model.name = "nova_carter_mpcc"
        
        x = ca.SX.sym("x", 3)
        u = ca.SX.sym("u", 2)   
        dt = ca.SX.sym("dt", 1)
        N = ca.SX.sym()

        self.ocp.model.disc_dyn_expr = ca.vertcat(
            x[0] + u[0] * np.cos(x[2]) * dt,
            x[1] + u[0] * np.sin(x[2]) * dt,
            x[2] + u[1] * dt,
        )
        self.ocp.dims.N = N
        self.ocp.model.x = x
        self.ocp.model.u = u