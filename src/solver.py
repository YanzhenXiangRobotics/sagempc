import timeit

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver
from botorch.models import SingleTaskGP

from src.utils.ocp import export_sempc_ocp

import matplotlib.pyplot as plt
from acados_template import AcadosOcp


class Plotter:
    def __init__(self, ocp: AcadosOcp, sqp_iters, Hm):
        self.plots_tmp = []
        self.scatters_tmp = []
        self.three_D_tmp = []

        self.ocp = ocp
        self.sqp_iters = sqp_iters

        self.fig, self.ax = plt.subplots()
        self.fig_3D = plt.figure()
        self.ax_3D = self.fig_3D.add_subplot(111, projection="3d")

        self.nx = ocp.model.x.shape[0] - 1
        self.nu = ocp.model.u.shape[0] - 1
        self.Hm = Hm

        self.create_grids()

        self.model_X_sim = []

    def create_grids(self):
        self.global_bbox = [-14.45293, 9.54707, -16.74178, 22.0763]
        self.resolution = 0.3
        self.X1_1D = np.arange(
            self.global_bbox[0], self.global_bbox[1], self.resolution
        )
        self.X2_1D = np.arange(
            self.global_bbox[2], self.global_bbox[3], self.resolution
        )
        self.X1, self.X2 = np.meshgrid(self.X1_1D, self.X2_1D)
        self.grids_list = np.vstack([self.X1.ravel(), self.X2.ravel()]).T

    def plot_goal(self, xg):
        self.ax.scatter(xg[0], xg[1], marker="*")

    def plot_gp(self, model: SingleTaskGP, type="p"):
        with torch.no_grad():
            pred = model(torch.tensor(self.grids_list, dtype=torch.float))
        if type == "p":
            y_plot = pred.mean - 2 * torch.sqrt(pred.variance)
        elif type == "o":
            y_plot = pred.mean + 2 * torch.sqrt(pred.variance)
        y_plot = y_plot.reshape(len(self.X2_1D), len(self.X1_1D)).detach().numpy()
        self.three_D_tmp.append(
            self.ax.contour(self.X1, self.X2, y_plot, levels=[0], colors="blue")
        )
        self.three_D_tmp.append(
            self.ax_3D.plot_surface(self.X1, self.X2, y_plot, color="orange", alpha=0.5)
        )

    def plot_sqp_sol(self, X_sol):
        for i in range(X_sol.shape[0] - 1):
            self.scatters_tmp.append(
                self.ax.arrow(
                    X_sol[i, 0],
                    X_sol[i, 1],
                    X_sol[i + 1, 0] - X_sol[i, 0],
                    X_sol[i + 1, 1] - X_sol[i, 1],
                    color="black",
                )
            )
        self.scatters_tmp.append(
            self.ax.scatter(
                X_sol[self.Hm, 0],
                X_sol[self.Hm, 1],
                s=5,
                marker="x",
                c="red",
            )
        )

    def plot_safe_set(self, X_sol, lb, dlb_dx):
        safe = torch.full((self.grids_list.shape[0],), True)
        for k in range(self.ocp.dims.N):
            lb_lin = np.zeros(self.grids_list.shape[0],)
            for i, x in enumerate(self.grids_list):
                lb_lin[i] = lb[k] + dlb_dx[k].T @ (x - X_sol[k, : self.nx - 1])
                if lb_lin[i] < 0:
                    safe[i] = False
            # lb_lin_2D = lb_lin.reshape(len(self.X2_1D), len(self.X1_1D))
            # self.ax_3D.plot_surface(self.X1, self.X2, lb_lin_2D)
        # plt.savefig("test_3D.png")

        self.scatters_tmp.append(
            self.ax.scatter(
                self.grids_list[safe, 0],
                self.grids_list[safe, 1],
                alpha=0.5,
                color="orange",
            )
        )

    def remove_plots(self):
        plots_tmp_len = len(self.plots_tmp)
        scatters_tmp_len = len(self.scatters_tmp)
        contours_tmp_len = len(self.three_D_tmp)
        for _ in range(plots_tmp_len):
            print(self.plots_tmp)
            self.plots_tmp.pop(0).pop(0).remove()
        for _ in range(scatters_tmp_len):
            self.scatters_tmp.pop(0).set_visible(False)
        for _ in range(contours_tmp_len):
            self.three_D_tmp.pop(0).remove()

    def plot_sqp(self, i, j, X_sol, lb, dlb_dx, X_sol_next):
        self.plot_safe_set(X_sol, lb, dlb_dx)
        self.plot_sqp_sol(X_sol_next)
        self.ax.axis("equal")
        self.ax.set_xlim([self.global_bbox[0], self.global_bbox[1]])
        self.ax.set_ylim([self.global_bbox[2], self.global_bbox[3]])
        self.fig.savefig(f"sqp_{i}_{j}.png")
        self.fig_3D.savefig(f"sqp_3D_{i}_{j}.png")
        if j != (self.sqp_iters - 1):
            self.remove_plots()

    def plot_sim(self, i, x_curr):
        self.model_X_sim.append(x_curr[:-1])
        model_X_sim = np.array(self.model_X_sim)
        self.ax.scatter(
            model_X_sim[:, 0],
            model_X_sim[:, 1],
            s=15,
            c="red",
        )
        self.ax.axis("equal")
        self.ax.set_xlim([self.global_bbox[0], self.global_bbox[1]])
        self.ax.set_ylim([self.global_bbox[2], self.global_bbox[3]])
        plt.savefig(f"sagempc_{i}.png")
        self.remove_plots()

# The class below is an optimizer class,
# it takes in GP function, x_g and rest are parameters
class SEMPC_solver(object):
    def __init__(self, params) -> None:
        ocp = export_sempc_ocp(params)
        self.name_prefix = (
            params["algo"]["type"]
            + "_env_"
            + str(params["env"]["name"])
            + "_i_"
            + str(params["env"]["i"])
            + "_"
        )
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file=self.name_prefix + "acados_ocp_sempc.json"
        )
        self.ocp_solver.store_iterate(self.name_prefix + "ocp_initialization.json")

        # sim = export_sim(params, 'sim_sempc')
        # self.sim_solver = AcadosSimSolver(
        #     sim, json_file='acados_sim_sempc.json')
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.max_sqp_iter = params["optimizer"]["SEMPC"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["SEMPC"]["tol_nlp"]
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]
        self.eps = params["common"]["epsilon"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        self.u_dim = params["optimizer"]["u_dim"]
        self.params = params
        self.state_dim = self.n_order * self.x_dim

        self.plotter = Plotter(self.ocp_solver.acados_ocp, self.max_sqp_iter, self.Hm)

    def initilization(self, sqp_iter, x_h, u_h):
        for stage in range(self.H):
            # current stage values
            x_h[stage, :] = self.ocp_solver.get(stage, "x")
            u_h[stage, :] = self.ocp_solver.get(stage, "u")
        x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
        if sqp_iter == 0:
            x_h_old = x_h.copy()
            u_h_old = u_h.copy()
            # initialize the first SQP iteration.
            for stage in range(self.H):
                if stage < (self.H - self.Hm):
                    # current stage values
                    x_init = x_h_old[stage + self.Hm, :].copy()
                    u_init = u_h_old[stage + self.Hm, :].copy()
                    x_init[-1] = (
                        x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]
                    ).copy()
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
                    half_time = x_init[-1].copy()
                else:
                    dt = (1.0 - half_time) / self.Hm
                    x_init = x_h_old[self.H, :].copy()  # reached the final state
                    x_init[-1] = half_time + dt * (stage - self.Hm)
                    z_init = x_init[0 : self.x_dim]
                    if (
                        self.params["algo"]["type"] == "ret_expander"
                        or self.params["algo"]["type"] == "MPC_expander"
                    ):
                        u_init = np.concatenate([np.array([0.0, 0.0, dt]), z_init])
                    else:
                        u_init = np.array([0.0, 0.0, dt])
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
            self.ocp_solver.set(self.H, "x", x_init)
            x_init[-1] = half_time + dt * (self.H - self.Hm)
            x_h[self.H, :] = x_init.copy()

        return x_h, u_h

    def path_init(self, path):
        split_path = np.zeros((self.H + 1, self.x_dim))
        interp_h = np.arange(self.Hm)
        path_step = np.linspace(0, self.Hm, path.shape[0])
        x_pos = np.interp(interp_h, path_step, path.numpy()[:, 0])
        y_pos = np.interp(interp_h, path_step, path.numpy()[:, 1])
        split_path[: self.Hm, 0], split_path[: self.Hm, 1] = x_pos, y_pos
        split_path[self.Hm :, :] = (
            np.ones_like(split_path[self.Hm :, :]) * path[-1].numpy()
        )
        # split the path into horizons
        for stage in range(self.H + 1):
            x_init = self.ocp_solver.get(stage, "x")
            x_init[: self.x_dim] = split_path[stage]
            self.ocp_solver.set(stage, "x", x_init)

    def solve(self, player, sample_iter):
        x_h = np.zeros((self.H + 1, self.state_dim + 1))
        u_h = np.zeros((self.H, self.u_dim + 1))  # u_dim

        w = 1e-3 * np.ones(self.H + 1)
        we = 1e-8 * np.ones(self.H + 1)
        we[int(self.H - 1)] = 10000
        # w[:int(self.Hm)] = 1e-1*np.ones(self.Hm)
        w[int(self.Hm)] = self.params["optimizer"]["w"]
        cw = 1e3 * np.ones(self.H + 1)
        if not player.goal_in_pessi:
            cw[int(self.Hm)] = 1
        xg = np.ones((self.H + 1, self.x_dim - 1)) * np.array(
            self.params["env"]["goal_loc"]
        )
        x_origin = player.origin[: self.x_dim].numpy()
        x_terminal = np.zeros(self.state_dim)
        x_terminal[: self.x_dim] = np.ones(self.x_dim) * x_origin
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set("rti_phase", 1)
            if (
                self.params["algo"]["type"] == "ret"
                or self.params["algo"]["type"] == "ret_expander"
            ):
                if player.goal_in_pessi:
                    x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
                else:
                    for stage in range(self.H):
                        # current stage values
                        x_h[stage, :] = self.ocp_solver.get(stage, "x")
                        u_h[stage, :] = self.ocp_solver.get(stage, "u")
                    x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
            else:
                #    pass
                x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
                if self.params["algo"]["init"] == "discrete":
                    self.path_init(player.solver_init_path)

            gp_val, gp_grad = player.get_gp_sensitivities(
                x_h[:, : self.x_dim - 1], "LB", "Cx"
            )  # pessimitic safe location
            UB_cx_val, UB_cx_grad = player.get_gp_sensitivities(
                x_h[:, : self.x_dim - 1], "UB", "Cx"
            )  # optimistic safe location
            # print(self.ocp_solver.acados_ocp.model.p)
            # stage = 0
            # t = [
            #     gp_val[stage],
            #     gp_grad[stage],
            #     x_h[stage, : self.x_dim - 1],
            #     xg[stage, :],
            #     w[stage],
            #     x_terminal,
            #     UB_cx_val[stage],
            #     UB_cx_grad[stage],
            #     cw[stage],
            # ]
            # print(t)
            # print("\n\n\n\n\n\n\n")
            for stage in range(self.H + 1):
                self.ocp_solver.set(
                    stage,
                    "p",
                    np.hstack(
                        (
                            gp_val[stage],
                            gp_grad[stage],
                            x_h[stage, : self.x_dim - 1],
                            xg[stage, :],
                            w[stage],
                            x_terminal,
                            UB_cx_val[stage],
                            UB_cx_grad[stage],
                            cw[stage],
                        )
                    ),
                )
            status = self.ocp_solver.solve()

            self.ocp_solver.options_set("rti_phase", 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            # self.ocp_solver.print_statistics()
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()

            if sqp_iter == 0:
                X_old = x_h.copy()
            else:
                X_old = X.copy()
            X, U, Sl = self.get_solution()
            self.plotter.plot_sqp(sample_iter, sqp_iter, X_old, gp_val, gp_grad, X)
            # print(X)
            # for stage in range(self.H):
            #     print(stage, " constraint ", self.constraint(LB_cz_val[stage], LB_cz_grad[stage], U[stage,3:5], X[stage,0:4], u_h[stage,-self.x_dim:], x_h[stage, :self.state_dim], self.params["common"]["Lc"]))
            # print("statistics", self.ocp_solver.get_stats("statistics"))
            if max(residuals) < self.tol_nlp:
                print("Residual less than tol", max(residuals), " ", self.tol_nlp)
                break
            if self.ocp_solver.status != 0:
                print(
                    "acados returned status {} in closed loop solve".format(
                        self.ocp_solver.status
                    )
                )
                self.ocp_solver.reset()
                self.ocp_solver.load_iterate(
                    self.name_prefix + "ocp_initialization.json"
                )

    def constraint(self, lb_cz_lin, lb_cz_grad, model_z, model_x, z_lin, x_lin, Lc):
        x_dim = self.x_dim
        tol = 1e-5
        ret = (
            lb_cz_lin
            + lb_cz_grad.T @ (model_z - z_lin)
            - (Lc / (ca.norm_2(x_lin[:x_dim] - z_lin) + tol))
            * ((x_lin[:x_dim] - z_lin).T @ (model_x - x_lin)[:x_dim])
            - (Lc / (ca.norm_2(x_lin[:x_dim] - z_lin) + tol))
            * ((z_lin - x_lin[:x_dim]).T @ (model_z - z_lin))
            - Lc * ca.norm_2(x_lin[:x_dim] - z_lin)
        )
        # ret = lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - 2*Lc*(x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim] - 2*Lc*(z_lin-x_lin[:x_dim]).T@(model_z-z_lin) - Lc*(x_lin[:x_dim] - z_lin).T@(x_lin[:x_dim] - z_lin)
        return ret, lb_cz_lin + lb_cz_grad.T @ (model_z - z_lin)

    def get_solution(self):
        X = np.zeros((self.H + 1, self.nx))
        U = np.zeros((self.H, self.nu))
        Sl = np.zeros((self.H + 1))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")
            # Sl[i] = self.ocp_solver.get(i, "sl")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U, Sl


class PytorchGPEvaluator1D(ca.Callback):
    def __init__(self, t_out, func, get_sparsity_out, opts={}):
        """
        t_in: list of inputs (pytorch tensors)
        t_out: list of outputs (pytorch tensors)
        """
        ca.casadi.Callback.__init__(self)
        self.t_out = t_out  # it is a function
        self.func = func
        self.get_sparsity_out = get_sparsity_out
        self.construct("PytorchGPEvaluator1D", opts)
        self.refs = []

    def update_func(self, function):
        self.func = function
        self.t_out = function  # it is a function

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(1, 1)

    def eval(self, x):
        y = torch.Tensor(x[0].toarray().tolist()[0] + [-2.0])
        # print("eval", y, [self.t_out(y.reshape(-1, 2)).detach().numpy()])
        return [self.t_out(y.reshape(-1, 2)).detach().numpy()]

    # def eval(self, x):
    #     y = torch.Tensor(x[0].toarray().tolist()[0])
    #     return [self.t_out(y.reshape(-1, 1)).detach().numpy()]

    def gradjac(self, x):
        jacob = torch.autograd.functional.jacobian(self.func, x)[0][0]
        return jacob.reshape(-1).detach()

    def gradhes(self, x):
        jacob = torch.autograd.functional.hessian(self.func, x)[0][0][0][0]
        return jacob.reshape(-1).detach()

    def jacob_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 1)

    def hessian_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 1)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # In the callback of jacobian we change to hessian functions
        callback = PytorchGPEvaluator1D(
            self.gradjac, self.func, self.jacob_sparsity_out
        )
        callback.jacob_sparsity_out = self.hessian_sparsity_out
        callback.gradjac = self.gradhes

        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out form that Cas.casadiexpects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        return ca.Function(
            name, nominal_in + nominal_out, callback.call(nominal_in), inames, onames
        )
