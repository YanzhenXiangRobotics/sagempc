import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_sim import AcadosSim

import sys, os

dir_here = os.path.abspath(os.path.dirname(__file__))
dir_project = os.path.join(dir_here, "..", "..")

sys.path.append(dir_project)

from src.utils.model import (
    export_nova_carter_discrete,
)


def concat_const_val(ocp, params):
    x_dim = params["optimizer"]["x_dim"]
    if (
        params["algo"]["type"] == "ret_expander"
        or params["algo"]["type"] == "MPC_expander"
    ):
        lbx = np.array(params["optimizer"]["x_min"])[:x_dim]
        ubx = np.array(params["optimizer"]["x_max"])[:x_dim]
        ocp.constraints.lbu = np.concatenate(
            [ocp.constraints.lbu, np.array([params["optimizer"]["dt"]]), lbx]
        )
        ocp.constraints.ubu = np.concatenate(
            [ocp.constraints.ubu, np.array([1.0]), ubx]
        )
        ocp.constraints.idxbu = np.arange(ocp.constraints.idxbu.shape[0] + 1 + x_dim)
    else:
        ocp.constraints.lbu = np.concatenate(
            [ocp.constraints.lbu, np.array([params["optimizer"]["dt"]])]
        )
        ocp.constraints.ubu = np.concatenate([ocp.constraints.ubu, np.array([1.0])])
        ocp.constraints.idxbu = np.concatenate(
            [ocp.constraints.idxbu, np.array([ocp.model.u.shape[0] - 1])]
        )

    # ocp.constraints.x0 = np.concatenate(
    #     [ocp.constraints.x0, np.array([0.0])])

    ocp.constraints.lbx_e = np.concatenate([ocp.constraints.lbx_e, np.array([1.0])])
    ocp.constraints.ubx_e = np.concatenate([ocp.constraints.ubx_e, np.array([1.0])])
    ocp.constraints.idxbx_e = np.concatenate(
        [ocp.constraints.idxbx_e, np.array([ocp.model.x.shape[0] - 1])]
    )

    ocp.constraints.lbx = np.concatenate([ocp.constraints.lbx, np.array([0])])
    ocp.constraints.ubx = np.concatenate([ocp.constraints.ubx, np.array([2])])
    ocp.constraints.idxbx = np.concatenate(
        [ocp.constraints.idxbx, np.array([ocp.model.x.shape[0] - 1])]
    )
    return ocp


def sempc_const_expr(model, x_dim, n_order, params, model_x_trans):
    lb_cx_lin = ca.SX.sym("lb_cx_lin")
    lb_cx_grad = ca.SX.sym("lb_cx_grad", x_dim - 1, 1)
    ub_cx_lin = ca.SX.sym("ub_cx_lin")
    ub_cx_grad = ca.SX.sym("ub_cx_grad", x_dim - 1, 1)

    x_lin = ca.SX.sym("x_lin", n_order * (x_dim - 1))
    x_terminal = ca.SX.sym("x_terminal", n_order * x_dim)

    xg = ca.SX.sym("xg", x_dim - 1)
    w = ca.SX.sym("w", 1, 1)
    cw = ca.SX.sym("cw", 1, 1)

    q_th = params["common"]["constraint"]
    var = (
        ub_cx_lin
        + ub_cx_grad.T @ (model_x_trans - x_lin)
        - (lb_cx_lin + lb_cx_grad.T @ (model_x_trans - x_lin))
    )
    p_lin = ca.vertcat(
        lb_cx_lin, lb_cx_grad, x_lin, xg, w, x_terminal, ub_cx_lin, ub_cx_grad, cw
    )
    model.con_h_expr = ca.vertcat(
        lb_cx_lin + lb_cx_grad.T @ (model_x_trans - x_lin) - q_th, cw * var
    )
    model.con_h_expr_e = ca.vertcat(
        lb_cx_lin + lb_cx_grad.T @ (model_x_trans - x_lin) - q_th
    )

    model.p = p_lin
    return model, w, xg, var


def sempc_cost_expr(ocp, model_x_trans, model_u, x_dim, w, xg, var, params):
    q = 1e-3 * np.diag(np.ones(model_u.shape[0]))
    qx = np.diag(np.ones(x_dim - 1))
    # cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (
        w * (model_x_trans - xg).T @ qx @ (model_x_trans - xg)
        + model_u.T @ (q) @ model_u
        + ocp.model.x[-1] * w / 1000
    )
    ocp.model.cost_expr_ext_cost_e = (
        w * (model_x_trans - xg).T @ qx @ (model_x_trans - xg)
    )

    ocp.constraints.idxsh = np.array([1])
    ocp.cost.zl = 1e2 * np.array([1])
    ocp.cost.zu = 1e1 * np.array([1])
    ocp.cost.Zl = 1e1 * np.array([1])
    ocp.cost.Zu = 1e1 * np.array([1])

    return ocp


def sempc_const_val(ocp, params, x_dim, u_dim):
    # constraints
    eps = params["common"]["epsilon"]  # - 0.05

    ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
    ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
    ocp.constraints.idxbu = np.arange(u_dim)

    lbx = np.array(params["optimizer"]["x_min"])
    ubx = np.array(params["optimizer"]["x_max"])

    # lbx = params["optimizer"]["u_min"][0]*np.ones(n_order*x_dim)
    # lbx[:x_dim] = params["optimizer"]["x_min"]*np.ones(x_dim)

    # ubx = params["optimizer"]["u_max"][0]*np.ones(n_order*x_dim)
    # ubx[:x_dim] = params["optimizer"]["x_max"]*np.ones(x_dim)

    x0 = np.zeros(ocp.model.x.shape[0])
    x0[:x_dim] = np.array(params["env"]["start_loc"])  # np.ones(x_dim)*0.72
    ocp.constraints.x0 = x0.copy()

    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])
    if params["algo"]["type"] == "MPC_Xn":
        wee = 1.0e-5
        ocp.constraints.lh = np.array([0, eps, -1 * wee, -1 * wee, -1 * wee, -1 * wee])
        ocp.constraints.uh = np.array([10.0, 1e8, wee, wee, wee, wee])
    elif (
        params["algo"]["type"] == "ret_expander"
        or params["algo"]["type"] == "MPC_expander"
    ):
        ocp.constraints.lh = np.array([0, eps, -1e8])
        ocp.constraints.uh = np.array([10.0, 1e8, 0.2])
    else:
        ocp.constraints.lh = np.array([0, eps])
        ocp.constraints.uh = np.array([10.0, 1e8])
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([10.0])

    # ocp.constraints.lh = np.array([0, eps])
    # ocp.constraints.uh = np.array([10.0, 1.0e9])
    # lh_e = np.zeros(n_order*x_dim+1)
    # lh_e[0] = 0
    # ocp.constraints.lh_e = lh_e
    # uh_e = np.zeros(n_order*x_dim+1)
    # uh_e[0] = 10
    # ocp.constraints.uh_e = uh_e

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp


def sempc_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["Tf"]

    ocp.solver_options.qp_solver_warm_start = 1
    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = 1.0e-1
    ocp.solver_options.integrator_type = "DISCRETE"  #'IRK'  # IRK
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    # ocp.solver_options.tol = 1e-6
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.globalization = 'FIXED_STEP'
    # ocp.solver_options.alpha_min = 1e-2
    # ocp.solver_options.__initialize_t_slacks = 0
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.levenberg_marquardt = 1e-1
    # ocp.solver_options.print_level = 2
    # ocp.solver_options.qp_solver_iter_max = 400
    # ocp.solver_options.regularize_method = 'MIRROR'
    # ocp.solver_options.exact_hess_constr = 0
    # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
    # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
    # ocp.solver_options.eps_sufficient_descent = 5e-1
    # params = {'globalization': ['MERIT_BACKTRACKING', 'FIXED_STEP'],
    #       'line_search_use_sufficient_descent': [0, 1],
    #       'globalization_use_SOC': [0, 1]}
    return ocp


def export_sempc_ocp(params):
    ocp = AcadosOcp()
    name_prefix = (
        params["algo"]["type"]
        + "_env_"
        + str(params["env"]["name"])
        + "_i_"
        + str(params["env"]["i"])
        + "_"
    )
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    u_dim = params["optimizer"]["u_dim"]
    # model = export_integrator_model('sempc')
    # model = export_n_integrator_model('sempc', n_order, x_dim)

    model = export_nova_carter_discrete(x_dim, u_dim)
    model_u = model.u[:-1]
    model_x_trans = model.x[:-2]

    model, w, xg, var = sempc_const_expr(model, x_dim, n_order, params, model_x_trans)

    ocp.model = model

    ocp = sempc_cost_expr(ocp, model_x_trans, model_u, x_dim, w, xg, var, params)

    ocp = sempc_const_val(ocp, params, x_dim, u_dim)

    ocp = concat_const_val(ocp, params)

    ocp = sempc_set_options(ocp, params)

    return ocp