import timeit

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver, AcadosSimSolver

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.ocp import export_oracle_ocp, export_sempc_ocp, export_sim

from geometry_msgs.msg import Twist
import time


# The class below is an optimizer class,
# it takes in GP function, x_g and rest are parameters
class SEMPC_solver(object):
    def __init__(
        self, params, grids_coupled, ax, legend_handles, fig, visu, fig_dir, publisher
    ) -> None:
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
        self.params = params
        if params["agent"]["dynamics"] == "robot":
            self.state_dim = self.n_order * self.x_dim + 1
        elif params["agent"]["dynamics"] == "nova_carter":
            self.state_dim = self.n_order * self.x_dim + 1
        else:
            self.state_dim = self.n_order * self.x_dim
        self.grids_coupled = grids_coupled
        self.visu = visu
        self.ax = ax
        self.legend_handles = legend_handles
        self.fig = fig
        self.plot_tmps, self.scatter_tmps, self.threeD_tmps = [], [], []
        self.fig_3D = plt.figure()
        self.ax_3D = self.fig_3D.add_subplot(111, projection="3d")
        self.fig_dir = fig_dir
        self.plot_per_sqp_iter = params["experiment"]["plot_per_sqp_iter"]
        self.publisher = publisher

        self.last_X = np.zeros((self.H + 1, self.state_dim + self.x_dim + 1))
        self.last_U = np.zeros((self.H, self.x_dim + 1 + self.x_dim))
        for stage in range(self.H):
            # current stage values
            self.last_X[stage, :] = self.ocp_solver.get(stage, "x")
            self.last_U[stage, :] = self.ocp_solver.get(stage, "u")
        self.last_X[self.H, :] = self.ocp_solver.get(self.H, "x")

        self.debug = self.params["experiment"]["debug"]
        self.time_ckp = time.time()
        self.local_plot_radius = 0.3
        self.z_initialized = False

    def update_x_curr(self, x_curr):
        self.x_curr = x_curr

    def initilization(self, sqp_iter, x_h, u_h):
        for stage in range(self.H):
            # current stage values
            x_h[stage, :] = self.ocp_solver.get(stage, "x")
            u_h[stage, :] = self.ocp_solver.get(stage, "u")
        x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
        if sqp_iter == 0:
            if self.params["optimizer"]["Hm"] == self.params["optimizer"]["H"]:
                u_h_old = u_h.copy()
                t = 0.0
                for stage in range(self.H):
                    x_h[stage, :] = np.concatenate(
                        (self.x_curr, np.zeros(self.x_dim), np.array([t]))
                    )
                    self.ocp_solver.set(stage, "x", x_h[stage, :])
                    u_h[stage, :] = np.concatenate(
                        (
                            np.zeros(self.x_dim),
                            np.array([self.params["optimizer"]["dt"]]),
                            u_h_old[-1, -self.x_dim :],
                        )
                    )
                    self.ocp_solver.set(stage, "u", u_h[stage, :])
                    t += self.params["optimizer"]["dt"]
                x_h[-1, :] = np.concatenate(
                    (self.x_curr, np.zeros(self.x_dim), np.array([t]))
                )
                self.ocp_solver.set(self.H, "x", x_h[-1, :])
            else:
                x_h_old = x_h.copy()
                u_h_old = u_h.copy()
                if (
                    self.params["algo"]["type"] == "ret_expander"
                    or self.params["algo"]["type"] == "MPC_expander"
                    or self.params["algo"]["type"] == "MPC_expander_V0"
                ):
                    u_h_old[:, -self.x_dim :] = x_h_old[:-1, : self.x_dim].copy()
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
                        half_time = x_init[-1].copy() + u_init[2]
                    else:
                        dt = (self.params["optimizer"]["Tf"] - half_time) / self.Hm
                        x_init = x_h_old[self.H, :].copy()  # reached the final state
                        x_init[-1] = half_time + dt * (stage - (self.H - self.Hm))
                        z_init = x_init[: self.x_dim]
                        # if not self.z_initialized:
                        #     z_init = x_init[: self.x_dim]
                        # else:
                        #     z_init = u_init[-self.x_dim :]
                        if (
                            self.params["algo"]["type"] == "ret_expander"
                            or self.params["algo"]["type"] == "MPC_expander"
                            or self.params["algo"]["type"] == "MPC_expander_V0"
                        ):
                            u_init = np.concatenate([np.array([0.0, 0.0, dt]), z_init])
                        else:
                            u_init = np.array([0.0, 0.0, dt])
                        self.ocp_solver.set(stage, "x", x_init)
                        self.ocp_solver.set(stage, "u", u_init)
                        x_h[stage, :] = x_init.copy()
                        u_h[stage, :] = u_init.copy()
                self.ocp_solver.set(self.H, "x", x_init)
                x_init[-1] = half_time + dt * self.Hm
                x_h[self.H, :] = x_init.copy()
                # x0 = np.zeros(self.state_dim)
                # x0[:self.x_dim] = np.ones(self.x_dim)*0.72
                # x0=np.concatenate([x0, np.array([0.0])])
                # x_init=x0.copy()
                # # x_init = self.ocp_solver.get(0, "x")
                # u_init = self.ocp_solver.get(0, "u")
                # Ts = 1/200
                # # MPC controller
                # x_init = np.array([0.72,0.72,0.0,0.0, 0.0])
                # u_init = np.array([-0.2,-0.2, Ts])

                #     x_h[stage, :] = x_init
                #     u_h[stage, :] = u_init
                # x_h[self.H, :] = x_init
                # self.ocp_solver.set(self.H, "x", x_init)
            # if sqp_iter == 0:
            #     print("Diff: ", x_h[:-1, : self.x_dim] - u_h[:, -self.x_dim :])
        self.z_initialized = True
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

    def plot_3D(self, player):
        X1, X2 = self.visu.x.numpy(), self.visu.y.numpy()
        with torch.no_grad():
            pred = player.Cx_model(self.grids_coupled)
            lower_list = pred.mean - player.Cx_beta * 2 * torch.sqrt(pred.variance)
            lower = lower_list.reshape((X1.shape[0], X2.shape[1]))
            mean = pred.mean.reshape((X1.shape[0], X2.shape[1]))
            self.threeD_tmps.append(
                self.ax_3D.plot_surface(X1, X2, lower, color="orange", alpha=0.5)
            )
            pessi_contour = self.ax.contour(
                X1,
                X2,
                lower,
                levels=[self.params["common"]["constraint"]],
                colors="blue",
                linewidths=0.5,
            )
            self.threeD_tmps.append(pessi_contour)
            (artists,), _ = pessi_contour.legend_elements()
            artists.set_label("pessimistic contour")
            self.legend_handles.append(artists)
            # mean_contour = self.ax.contour(
            #     X1,
            #     X2,
            #     mean,
            #     levels=[self.params["common"]["constraint"]],
            #     colors="pink",
            #     linewidths=0.5,
            # )
            # self.threeD_tmps.append(mean_contour)
            # (artists,), _ = mean_contour.legend_elements()
            # artists.set_label("mean contour")
            # self.legend_handles.append(artists)
            # if (
            #     self.params["algo"]["type"] == "MPC_expander"
            #     or self.params["algo"]["type"] == "MPC_expander_V0"
            # ):
            #     self.plot_expander(lower_list)

    def plot_expander(self, lower_list):
        import math

        resolution = self.params["visu"]["step_size"]
        q_th = self.params["common"]["constraint"]
        lower_list -= q_th
        # candidates = self.grids_coupled[lower_list >= 0]
        candi_indices = torch.where(lower_list >= 0)[0]
        expander = set()
        for idx in candi_indices:
            i, j = (
                math.floor(idx / self.params["env"]["shape"]["y"]),
                idx % self.params["env"]["shape"]["y"],
            )
            expander.add((i, j))
            r = lower_list[idx]
            ext_dist = math.floor(r / resolution)
            for i_1 in np.arange(i - ext_dist, i + ext_dist + 1, dtype=int):
                for j_1 in np.arange(j - ext_dist, j + ext_dist + 1, dtype=int):
                    if np.linalg.norm(np.array([i - i_1, j - j_1])) <= ext_dist:
                        expander.add((i_1, j_1))
        expander_pos = []
        for idx_couple in expander:
            idx = idx_couple[0] * self.params["env"]["shape"]["y"] + idx_couple[1]
            pos = self.grids_coupled[idx].numpy()
            if lower_list[idx] <= self.params["common"]["expander_offset"]:
                expander_pos.append(pos)
        expander_pos = np.array(expander_pos)
        if expander_pos.size != 0:
            tmp = self.ax.scatter(
                expander_pos[:, 0],
                expander_pos[:, 1],
                c="orange",
                alpha=0.3,
                s=3,
                label="expander",
            )
            self.scatter_tmps.append(tmp)
            self.legend_handles.append(tmp)

    def plot_safe_set(self, gp_val, gp_grad, x_h):
        safe = torch.full((self.grids_coupled.shape[0],), True)
        for k in range(self.H + 1):
            for i, x in enumerate(self.grids_coupled.numpy()):
                const_val = (
                    gp_val[k]
                    + gp_grad[k].T @ (x - x_h[k, : self.x_dim])
                    - self.params["common"]["constraint"]
                )
                if (const_val < 0) or (const_val > 10.0):
                    safe[i] = False

    def log_duration(self, prefix="Time", last_time=None):
        duration = (
            time.time() - self.time_ckp
            if last_time is None
            else time.time() - last_time
        )
        print(f"{prefix}: {duration}")
        self.time_ckp = time.time()
        return self.time_ckp

    def plot_last_sol_shifted_sol(self, x_h, u_h, player, sim_iter):
        xh_plot, xh_m_plot, zh_m_plot = self.plot_sqp_sol(
            x_h, u_h[self.Hm, -self.x_dim :]
        )
        lastX_plot, lastX_m_plot = self.plot_sqp_sol(self.last_X, c="orange")
        self.plot_3D(player)
        self.ax.set_xlim(
            [
                self.x_curr[0] - self.local_plot_radius,
                self.x_curr[0] + self.local_plot_radius,
            ]
        )
        self.ax.set_ylim(
            [
                self.x_curr[1] - self.local_plot_radius,
                self.x_curr[1] + self.local_plot_radius,
            ]
        )
        if not os.path.exists("sqp_sols"):
            os.makedirs("sqp_sols")
        self.fig.savefig(os.path.join("sqp_sols", f"sol_{sim_iter-1}_shifted.png"))
        xh_plot.remove()
        xh_m_plot.set_visible(False)
        zh_m_plot.set_visible(False)
        lastX_plot.remove()
        lastX_m_plot.set_visible(False)

    def log_unsafe(self, X, U, player):
        LB_cz_val_next, _ = player.get_gp_sensitivities(U[:, -self.x_dim :], "LB", "Cx")
        GP_vals_next = LB_cz_val_next - np.linalg.norm(
            X[:-1, : self.x_dim] - U[:, -self.x_dim :], axis=-1
        )
        GP_val_next = GP_vals_next[self.Hm]
        if GP_val_next < self.params["common"]["constraint"]:
            print("Unsafe !!!")
            print(f"GP_val_next: {GP_val_next}")

    def plot_sol(self, X, U, sqp_iter, sim_iter):
        self.log_duration("Time before plotting sqp sol")
        if (
            self.params["algo"]["type"] == "ret_expander"
            or self.params["algo"]["type"] == "MPC_expander"
            or self.params["algo"]["type"] == "MPC_expander_V0"
        ):
            X_plot, Xm_plot, Zm_plot = self.plot_sqp_sol(X, U[self.Hm, -self.x_dim :])
            self.scatter_tmps.append(Zm_plot)

        else:
            X_plot, Xm_plot = self.plot_sqp_sol(sqp_iter, X)
        self.plot_tmps.append(X_plot)
        self.scatter_tmps.append(Xm_plot)
        self.ax.set_xlim(
            [
                self.x_curr[0] - self.local_plot_radius,
                self.x_curr[0] + self.local_plot_radius,
            ]
        )
        self.ax.set_ylim(
            [
                self.x_curr[1] - self.local_plot_radius,
                self.x_curr[1] + self.local_plot_radius,
            ]
        )
        if not os.path.exists("sqp_sols"):
            os.makedirs("sqp_sols")
        self.fig.savefig(os.path.join("sqp_sols", f"sol_{sim_iter}_{sqp_iter}.png"))
        if (sqp_iter == self.max_sqp_iter - 1) or self.early_term:
            len_plot_tmps = len(self.plot_tmps)
            len_scatter_tmps = len(self.scatter_tmps)
            for _ in range(len_plot_tmps):
                self.plot_tmps.pop(0).remove()
            for _ in range(len_scatter_tmps):
                self.scatter_tmps.pop(0).set_visible(False)

    def solve(self, player, sim_iter):
        x_h = np.zeros((self.H + 1, self.state_dim + self.x_dim + 1))
        if (
            self.params["algo"]["type"] == "ret_expander"
            or self.params["algo"]["type"] == "MPC_expander"
            or self.params["algo"]["type"] == "MPC_expander_V0"
        ):
            u_h = np.zeros((self.H, self.x_dim + 1 + self.x_dim))  # u_dim
        else:
            u_h = np.zeros((self.H, self.x_dim + 1))  # u_dim
        if self.debug:
            self.log_duration("Time between two solving")
        w = 1e-3 * np.ones(self.H + 1)
        we = 1e-8 * np.ones(self.H + 1)
        we[int(self.H - 1)] = 10000
        # w[:int(self.Hm)] = 1e-1*np.ones(self.Hm)
        w[int(self.Hm)] = self.params["optimizer"]["w"]
        cw = 1e3 * np.ones(self.H + 1)
        if not player.goal_in_pessi:
            cw[int(self.Hm)] = 1
        w_v_omega = 1e-3 * np.ones(self.H)
        w_v_omega[int(self.Hm) - 1] = 1e5
        xg = np.ones((self.H + 1, self.x_dim)) * player.get_next_to_go_loc()
        x_origin = player.origin[: self.x_dim].numpy()
        x_terminal = np.zeros(self.state_dim)
        x_terminal[: self.x_dim] = np.ones(self.x_dim) * x_origin
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set("rti_phase", 1)
            x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
            if self.debug and (sqp_iter == 0):
                self.plot_last_sol_shifted_sol(x_h, u_h, player, sim_iter)
            gp_val, gp_grad = player.get_gp_sensitivities(
                x_h[:, : self.x_dim], "LB", "Cx"
            )  # pessimitic safe location
            UB_cx_val, UB_cx_grad = player.get_gp_sensitivities(
                x_h[:, : self.x_dim], "UB", "Cx"
            )  # optimistic safe location
            if (
                self.params["algo"]["type"] == "ret_expander"
                or self.params["algo"]["type"] == "MPC_expander"
                or self.params["algo"]["type"] == "MPC_expander_V0"
            ):
                LB_cz_val, LB_cz_grad = player.get_gp_sensitivities(
                    u_h[:, -self.x_dim :], "LB", "Cx"
                )
                for stage in range(self.H):
                    self.ocp_solver.set(
                        stage,
                        "p",
                        np.hstack(
                            (
                                gp_val[stage],
                                gp_grad[stage],
                                x_h[stage, : self.state_dim],
                                xg[stage],
                                w[stage],
                                x_terminal,
                                UB_cx_val[stage],
                                UB_cx_grad[stage],
                                cw[stage],
                                u_h[stage, -self.x_dim :],
                                LB_cz_val[stage],
                                LB_cz_grad[stage],
                                w_v_omega[stage],
                            )
                        ),
                    )
                stage = self.H  # stage already is at self.H
                self.ocp_solver.set(
                    stage,
                    "p",
                    np.hstack(
                        (
                            gp_val[stage],
                            gp_grad[stage],
                            x_h[stage, : self.state_dim],
                            xg[stage],
                            w[stage],
                            x_terminal,
                            UB_cx_val[stage],
                            UB_cx_grad[stage],
                            cw[stage],
                            u_h[stage - 1, -self.x_dim :],
                            LB_cz_val[stage - 1],
                            LB_cz_grad[stage - 1],
                            w_v_omega[stage - 1],
                        )
                    ),
                )  # last 3 "stage-1" are dummy values
            elif self.params["algo"]["type"] == "MPC_Xn":
                for stage in range(self.H + 1):
                    self.ocp_solver.set(
                        stage,
                        "p",
                        np.hstack(
                            (
                                gp_val[stage],
                                gp_grad[stage],
                                x_h[stage, : self.state_dim],
                                xg[stage],
                                w[stage],
                                x_terminal,
                                UB_cx_val[stage],
                                UB_cx_grad[stage],
                                cw[stage],
                                we[stage],
                            )
                        ),
                    )
            else:
                for stage in range(self.H + 1):
                    self.ocp_solver.set(
                        stage,
                        "p",
                        np.hstack(
                            (
                                gp_val[stage],
                                gp_grad[stage],
                                x_h[stage, : self.state_dim],
                                xg[stage],
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
            # print("cost res", self.ocp_solver.get_cost(), self.ocp_solver.get_residuals())

            X_raw, U_raw, Sl = self.get_solution()
            if self.debug:
                ckp = self.log_duration("Time solving problem")
            # print(
            #     "cost ",
            #     self.ocp_solver.get_cost(),
            #     "cost Xm ",
            #     (X_raw[self.Hm, : self.x_dim] - xg[self.Hm]).T
            #     @ (X_raw[self.Hm, : self.x_dim] - xg[self.Hm]),
            #     "cost Um ",
            #     1e-3 * U_raw[self.Hm, : self.x_dim].T @ U_raw[self.Hm, : self.x_dim],
            #     "cost Tm ",
            #     X_raw[self.Hm, -1] / 1000,
            #     "cost step ",
            #     np.sum(
            #         [
            #             0.1
            #             * (
            #                 (X_raw[k, : self.x_dim] - x_h[k, : self.x_dim]).T
            #                 @ (X_raw[k, : self.x_dim] - x_h[k, : self.x_dim])
            #                 + (U_raw[k, -self.x_dim :] - u_h[k, -self.x_dim :]).T
            #                 @ (U_raw[k, -self.x_dim :] - u_h[k, -self.x_dim :])
            #             )
            #             for k in range(self.H)
            #         ]
            #     ),
            # )

            if self.params["common"]["backtrack"]:
                X, U, alpha = self.backtrack(
                    X_raw, U_raw, x_h, u_h, player, sqp_iter, sim_iter
                )
            if self.debug:    
                self.log_duration("Time for backtracking", ckp)
            else:
                alpha = 1.0
                X, U = X_raw.copy(), U_raw.copy()
            max_step_size = np.max((np.max(abs(X - x_h)), np.max(abs(U - u_h))))
            if self.debug:
                print(
                    "Sim iter: ",
                    sim_iter,
                    "SQP iter: ",
                    sqp_iter,
                    "Alpha: ",
                    alpha,
                    "Max step size: ",
                    max_step_size,
                    "\nX: ",
                    X,
                    "\nU: ",
                    U,
                )

            self.set_solution(X, U)

            residuals = self.ocp_solver.get_residuals()
            if self.debug:
                self.log_unsafe(X, U, player)
            if max_step_size < 0.04:
                print(f"Break at sqp iter {sqp_iter}, \n\n")
                self.early_term = True
            else:
                self.early_term = False
            if self.debug:
                self.plot_sol(X, U, sqp_iter, sim_iter)

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
            if self.debug:
                self.log_duration("Time plotting sqp sol & save fig")
                sqp_plot_dir = os.path.join(self.fig_dir, f"sol_{sim_iter}")
                if not os.path.exists(sqp_plot_dir) and self.plot_per_sqp_iter:
                    os.makedirs(sqp_plot_dir)
                    self.fig.savefig(os.path.join(sqp_plot_dir, f"{sqp_iter}"))
            if self.debug:
                if sqp_iter == (self.max_sqp_iter - 1):
                    tmp_1, _ = player.get_gp_sensitivities(
                        X[:, : self.x_dim], "LB", "Cx"
                    )
                    tmp_2_vals, tmp_2_grads = player.get_gp_sensitivities(
                        self.last_X[:, : self.x_dim], "LB", "Cx"
                    )
                    step_size = (X - self.last_X)[:, : self.x_dim]
                    lin_gp_vals = []
                    for val, grad, x, x_lin in zip(
                        tmp_2_vals,
                        tmp_2_grads,
                        X[:, : self.x_dim],
                        self.last_X[:, : self.x_dim],
                    ):
                        lin_gp_val = val + grad @ (x - x_lin).T
                        lin_gp_vals.append(lin_gp_val)
                    tmp_2_vals = np.array(lin_gp_vals)
                    # print(f"Before updating GP, gp val: {tmp_1}, lin gp val: {lin_gp_vals}")
            self.last_X, self.last_U = X.copy(), U.copy()
            if self.early_term:
                break
        return X, U

    def backtrack(self, X, U, x_h, u_h, player, sqp_iter, sim_iter):
        alpha = 1.0
        gp_val_next, _ = player.get_gp_sensitivities(X[:, : self.x_dim], "LB", "Cx")
        LB_cz_val_next, _ = player.get_gp_sensitivities(U[:, -self.x_dim :], "LB", "Cx")
        backtracking_printed = False
        Lc = self.params["common"]["Lc"]
        # while (
        #     any(
        #         LB_cz_val_next
        #         - Lc
        #         * np.linalg.norm(X[:-1, : self.x_dim] - U[:, -self.x_dim :], axis=-1)
        #         < self.params["common"]["constraint"]
        #     )
        # ) and (alpha > 0.02):
        # self.log_duration()
        while (
            any(
                LB_cz_val_next
                - Lc
                * np.linalg.norm(X[:-1, : self.x_dim] - U[:, -self.x_dim :], axis=-1)
                < self.params["common"]["constraint"]
            )
            or (gp_val_next[-1] < self.params["common"]["constraint"])
        ) and (alpha > 0.02):
            # while (any(gp_val_next < self.params["common"]["constraint"])) and (
            #     alpha >= 0.0
            # ):
            # if not backtracking_printed:
            #     if backtrack_H:
            #         print("backtrack_H")
            # print("Backtracking")
            backtracking_printed = True
            # print(f"Backtracking... alpha={alpha}")
            alpha -= 0.1
            X = alpha * X + (1 - alpha) * x_h
            U = alpha * U + (1 - alpha) * u_h
            # self.log_duration()
            gp_val_next, _ = player.get_gp_sensitivities(X[:, : self.x_dim], "LB", "Cx")
            # self.log_duration()
            LB_cz_val_next, _ = player.get_gp_sensitivities(
                U[:, -self.x_dim :], "LB", "Cx"
            )
            # self.log_duration()
        return X, U, alpha

    def compute_Lc_constr_next_lin(self, X, U, x_h, z_h, player):
        lb_cz_lins, lb_cz_grads = player.get_gp_sensitivities(z_h, "LB", "Cx")
        Lc = self.params["common"]["Lc"]
        q_th = self.params["common"]["constraint"]
        tol = self.params["common"]["Lc_lin_tol"]
        LB_cv_val_lins = []
        for model_x, model_z, x_lin, z_lin, lb_cz_lin, lb_cz_grad in zip(
            X, U, x_h, z_h, lb_cz_lins, lb_cz_grads
        ):
            LB_cv_val_lin = (
                lb_cz_lin
                + lb_cz_grad @ (model_z - z_lin).T
                - (Lc / (np.linalg.norm(x_lin - z_lin) + tol))
                * ((x_lin - z_lin) @ (model_x - x_lin).T)
                - (Lc / (np.linalg.norm(x_lin - z_lin) + tol))
                * ((z_lin - x_lin) @ (model_z - z_lin).T)
                - Lc * np.linalg.norm(x_lin - z_lin)
                - q_th,
            )
            LB_cv_val_lins.append(LB_cv_val_lin[0])
        LB_cv_val_lins = np.array(LB_cv_val_lins)
        return LB_cv_val_lins

    def plot_sqp_sol(self, X, zm=None, c="black"):
        (X_plot,) = self.ax.plot(X[:, 0], X[:, 1], c=c, linewidth=0.5)
        Xm_plot = self.ax.scatter(X[self.Hm, 0], X[self.Hm, 1], c=c, marker="x", s=30)
        X_plot.set_label("X solution")
        Xm_plot.set_label("X solution at Hm")
        self.legend_handles += [X_plot, Xm_plot]
        if zm is not None:
            Zm_plot = self.ax.scatter(zm[0], zm[1], c="cyan", marker="x", s=30)
            Zm_plot.set_label("Z solution at Hm")
            self.legend_handles.append(Zm_plot)
            return X_plot, Xm_plot, Zm_plot
        else:
            return X_plot, Xm_plot

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

    def model_ss(self, model_x):
        val = model_x - model.f_expl_expr[:-1]

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

    def set_solution(self, X, U):
        for i in range(self.H):
            self.ocp_solver.set(i, "x", X[i, :])
            self.ocp_solver.set(i, "u", U[i, :])
        self.ocp_solver.set(self.H, "x", X[self.H, :])

    def get_solver_status():
        return None


class Oracle_solver(object):
    def __init__(self, params) -> None:
        ocp = export_oracle_ocp(params)
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file=params["algo"]["type"] + "acados_ocp_oracle.json"
        )

        # sim = export_sim(params, 'sim_oracle')
        # self.sim_solver = AcadosSimSolver(
        #     sim, json_file='acados_sim_oracle.json')
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.max_sqp_iter = params["optimizer"]["oracle"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["oracle"]["tol_nlp"]
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]
        self.eps = params["common"]["epsilon"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        # should every player have its own solver?
        pass

    def solve(self, player):
        x_h = np.zeros((self.H + 1, self.state_dim + 1))
        u_h = np.zeros((self.H, self.x_dim + 1))
        w = 1e-3 * np.ones(self.H + 1)
        w[int(self.H / 2)] = 10
        xg = np.ones((self.H + 1, self.x_dim)) * player.get_utility_minimizer
        x_origin = player.origin[: self.x_dim].numpy()
        x_terminal = np.zeros(self.state_dim)
        x_terminal[: self.x_dim] = np.ones(self.x_dim) * x_origin
        # xg = player.opti_UCB()*np.ones((self.H+1, self.x_dim))
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set("rti_phase", 1)
            for stage in range(self.H):
                # current stage values
                x_h[stage, :] = self.ocp_solver.get(stage, "x")
                u_h[stage, :] = self.ocp_solver.get(stage, "u")
                # if sqp_iter<1:
                #     x_h[stage,0:2] += np.random.randint(-100,100, size=(2))/100000
            x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
            # if sqp_iter == 0:
            #     x_h_old = x_h.copy()
            #     u_h_old = u_h.copy()
            #     # initialize the first SQP iteration.
            #     for stage in range(self.H):
            #         if stage < self.Hm:
            #             # current stage values
            #             x_init = x_h_old[stage + self.Hm, :].copy()
            #             u_init = u_h_old[stage + self.Hm, :].copy()
            #             x_init[-1] = (x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]).copy()
            #             self.ocp_solver.set(stage, "x", x_init)
            #             self.ocp_solver.set(stage, "u", u_init)
            #             x_h[stage, :] = x_init.copy()
            #             u_h[stage, :] = u_init.copy()
            #             half_time = x_init[-1].copy()
            #         else:
            #             dt = (1-half_time)/self.Hm
            #             x_init = x_h_old[self.H, :].copy() # reached the final state
            #             x_init[-1] = half_time + dt*(stage-self.Hm)
            #             u_init = np.array([0.0,0.0, dt])
            #             self.ocp_solver.set(stage, "x", x_init)
            #             self.ocp_solver.set(stage, "u", u_init)
            #             x_h[stage, :] = x_init.copy()
            #             u_h[stage, :] = u_init.copy()
            #     self.ocp_solver.set(self.H, "x", x_init)
            #     x_init[-1] = half_time + dt*(self.H-self.Hm)
            #     x_h[self.H, :] = x_init.copy()
            # # print("x_h", x_h)
            # # eps_lin = self.ocp_solver.get(stage, "eps")
            UB_cx_val, UB_cx_grad = player.get_gp_sensitivities(
                x_h[:, : self.x_dim], "UB", "Cx"
            )  # optimistic safe location
            UB_cx_val -= self.eps
            # UB_fx_val, UB_fx_grad = player.get_gp_sensitivities(
            #     x_h[:, :self.x_dim], "UB", "Fx")
            # LB_cx_val, LB_cx_grad = player.get_gp_sensitivities(
            #     x_h[:, 0], "LB", "Cx")
            # UB_fx_grad = UB_cx_grad - LB_cx_grad

            for stage in range(self.H + 1):
                self.ocp_solver.set(
                    stage,
                    "p",
                    np.hstack(
                        (
                            UB_cx_val[stage],
                            UB_cx_grad[stage],
                            x_h[stage, : self.state_dim],
                            x_terminal,
                            xg[stage],
                            w[stage],
                        )
                    ),
                )
            # self.ocp_solver.set(int(self.H/2), "x",
            #                     player.planned_measure_loc[0].reshape(1, 1).numpy())
            status = self.ocp_solver.solve()
            #
            self.ocp_solver.options_set("rti_phase", 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            # self.ocp_solver.print_statistics()
            residuals = self.ocp_solver.get_residuals()
            if max(residuals) < self.tol_nlp:
                print("Residual less than tol", max(residuals), " ", self.tol_nlp)
                break

    def get_solution(self):
        X = np.zeros((self.H + 1, self.nx))
        U = np.zeros((self.H, self.nu))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U

    def get_solver_status():
        return None


class GoalOPT(object):
    def __init__(
        self,
        optim_param,
        common_param,
        agent_param,
        LB_const_eval,
        UB_const_eval,
        UB_obj_eval,
    ) -> None:
        self.H = optim_param["H"]
        self.optim_param = optim_param
        self.Hm = optim_param["Hm"]
        self.u_min = optim_param["u_min"]
        self.u_max = optim_param["u_max"]
        self.x_min = optim_param["x_min"]
        self.x_max = optim_param["x_max"]
        self.Lc = agent_param["Lc"]
        self.constraint = common_param["constraint"]
        self.epsQ = common_param["epsilon"]
        self.UB_const_eval = UB_const_eval
        self.LB_const_eval = LB_const_eval
        self.UB_obj_eval = UB_obj_eval
        self.formulation1D()
        pass

    def getx(self):
        return self.opti.value(self.x)

    def getu(self):
        return self.opti.value(self.u)

    def setstartparam(self, loc):
        self.opti.set_value(self.p_start, loc.tolist())
        self.opti.set_initial(self.x[0], loc.tolist())

    def setendparam(self, loc):
        self.opti.set_value(self.p_end, loc.tolist())
        self.opti.set_initial(self.x[1:], loc.tolist())
        # init_rand = loc + (torch.rand(self.H) - 0.5)
        # self.opti.set_initial(self.x[1:], init_rand.reshape(-1, 1).numpy())

    def get_candidate(self):
        return self.opti.value(self.x)[self.Hm + 1]

    def formulation1D(self):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(1, self.H + 1)
        self.u = self.opti.variable(1, self.H)
        self.p_start = self.opti.parameter(1)
        self.p_end = self.opti.parameter(1)

        self.opti.subject_to(self.x[1:] == self.x[:-1] + self.u[:])
        self.opti.subject_to(self.u[0:] <= self.u_max)
        self.opti.subject_to(self.u[0:] >= self.u_min)
        self.opti.subject_to(self.x[1:] <= self.x_max)
        self.opti.subject_to(self.x[1:] >= self.x_min)
        # self.opti.subject_to(self.UB_const_eval.call(
        #     [self.x[k+1]])[0] - self.epsQ >= self.constraint)
        self.opti.subject_to(
            ca.fmax(
                self.LB_const_eval.call([self.x[1:]])[0],
                self.UB_const_eval.call([self.x[1:]])[0] - self.epsQ,
            )
            >= self.constraint
        )

        # for k in range(0, self.H):
        #     self.opti.set_initial(self.x[k+1], self.p_start)
        #     self.opti.subject_to(self.x[k+1] == self.x[k] + self.u[k])
        #     self.opti.subject_to(self.u[k] <= self.u_max)
        #     self.opti.subject_to(self.u[k] >= self.u_min)
        #     self.opti.subject_to(self.x[k+1] <= self.x_max)
        #     self.opti.subject_to(self.x[k+1] >= self.x_min)
        #     # self.opti.subject_to(self.UB_const_eval.call(
        #     #     [self.x[k+1]])[0] - self.epsQ >= self.constraint)
        #     self.opti.subject_to(ca.fmax(self.LB_const_eval.call(
        #         [self.x[k+1]])[0], self.UB_const_eval.call(
        #         [self.x[k+1]])[0] - self.epsQ) >= self.constraint)

        self.opti.subject_to(self.x[0] == self.p_start)
        self.opti.subject_to(self.x[self.H] == self.p_end)
        # % And choose a concrete value for p, latter can be passed from another location
        self.opti.minimize(
            -10 * ca.sumsqr(self.UB_obj_eval.call([self.x[self.Hm + 1]])[0])
            + 0.001 * ca.sumsqr(self.u)
        )

    def solve(self):  # parameterize and then solve
        # p_opts = {"expand": True}
        # self.opti.print_options()
        s_opts = {
            "ipopt.max_iter": 10,
            "ipopt.tol": 1e-8,
            "ipopt.print_timing_statistics": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.linear_solver": self.optim_param["linear_solver"],
        }
        self.opti.solver("ipopt", s_opts)
        # self.opti.solver("sqpmethod")
        try:
            sol = self.opti.solve()
        except:
            print("did not converge, let see .debug value")
            self.opti = self.opti.debug
        # print("solver status", self.opti.stats()['success'])
        # print("solver status", self.opti.stats())

    def print(self):
        print(self.opti)


class SEMPC(object):
    def __init__(
        self, optim_param, common_param, agent_param, LB_eval, UB_eval
    ) -> None:
        self.optim_param = optim_param
        self.H = optim_param["H"]
        self.Hm = optim_param["Hm"]
        self.u_min = optim_param["u_min"]
        self.u_max = optim_param["u_max"]
        self.x_min = optim_param["x_min"]
        self.x_max = optim_param["x_max"]
        self.Lc = agent_param["Lc"]
        self.constraint = common_param["constraint"]
        self.epsQ = common_param["epsilon"]
        self.LB_eval = LB_eval
        self.UB_eval = UB_eval
        self.GPformulation1D()
        return None

    def update_const_func(self, LB, UB):
        self.LB = LB
        self.UB = UB

    def formulation(self, x_goal):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(2, self.H + 1)
        self.u = self.opti.variable(2, self.H)
        self.z = self.opti.variable(2, 1)

        p = self.opti.parameter(2, 1)

        for k in range(0, self.H):
            self.opti.subject_to(self.x[:, k + 1] == self.x[:, k] + self.u[:, k])
            # self.opti.subject_to(lb(self.x[k+1]) >= 2.25)
            self.opti.subject_to(self.u[:, k] <= self.u_max)
            self.opti.subject_to(self.u[:, k] >= self.u_min)
            self.opti.subject_to(self.x[:, k + 1] <= self.x_max)
            self.opti.subject_to(self.x[:, k + 1] >= self.x_min)
            # self.opti.subject_to(self.x[1, k+1] == -2)
        self.opti.subject_to(self.x[:, self.Hm] == self.z)
        self.opti.subject_to(self.x[:, 0] == self.p_start)
        self.opti.subject_to(self.x[:, self.H] == self.p_end)
        # % And choose a concrete value for p, latter can be passed from another location
        # self.opti.set_value(p, [0, -2])
        self.opti.minimize(
            ca.sumsqr(self.z - x_goal.numpy()) * 3 + 0.001 * ca.sumsqr(self.u)
        )

    def setparam(self, loc):
        self.p = loc.tolist()

    def setstartparam(self, loc):
        self.opti.set_value(self.p_start, loc.tolist())

    def setendparam(self, loc):
        self.opti.set_value(self.p_end, loc.tolist())

    def setgoalparam(self, loc):
        self.opti.set_value(self.x_goal_p, loc.tolist())
        self.opti.set_initial(self.z, [loc.numpy()])

    def getx(self):
        return self.opti.value(self.x)

    def getu(self):
        return self.opti.value(self.u)

    def getmin_constraint(self):
        return self.opti.value(self.u)

    def getz(self):
        return self.opti.value(self.z)

    def setwarmstartparam(self, x, u):
        self.warm_x = x
        self.warm_u = u
        self.opti.set_initial(self.x, self.warm_x)
        self.opti.set_initial(self.u, self.warm_u)

    def GPformulation1D(self):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(1, self.H + 1)
        self.u = self.opti.variable(1, self.H)
        self.z = self.opti.variable(1, 1)
        self.x_goal_p = self.opti.parameter(1)
        self.p_start = self.opti.parameter(1)
        self.p_end = self.opti.parameter(1)

        self.opti.subject_to(self.x[1:] == self.x[:-1] + self.u[:])
        self.opti.subject_to(self.u[0:] <= self.u_max)
        self.opti.subject_to(self.u[0:] >= self.u_min)
        self.opti.subject_to(self.x[1:] <= self.x_max)
        self.opti.subject_to(self.x[1:] >= self.x_min)
        self.opti.subject_to(self.LB_eval.call([self.x[1:]])[0] >= self.constraint)

        # for k in range(0, self.H):
        #     # self.opti.set_initial(self.x[k+1], self.p_start)
        #     self.opti.subject_to(self.x[k+1] == self.x[k] + self.u[k])
        #     self.opti.subject_to(self.u[k] <= self.u_max)
        #     self.opti.subject_to(self.u[k] >= self.u_min)
        #     self.opti.subject_to(self.x[k+1] <= self.x_max)
        #     self.opti.subject_to(self.x[k+1] >= self.x_min)
        #     self.opti.subject_to(self.LB_eval.call(
        #         [self.x[k+1]])[0] >= self.constraint)
        #     # self.opti.subject_to(self.UB_eval.call(
        #     #     [self.x[k+1]])[0] - self.epsQ >= self.constraint)

        # Informative points
        self.opti.subject_to(
            self.UB_eval.call([self.x[self.Hm + 1]])[0]
            - self.LB_eval.call([self.x[self.Hm + 1]])[0]
            >= self.epsQ
        )
        # Expander condition
        self.opti.subject_to(
            self.UB_eval.call([self.x[self.Hm + 1]])[0]
            - ca.mtimes(ca.DM([self.Lc]), ca.norm_2(self.x[self.Hm + 1] - self.z))
            >= self.constraint
        )

        self.opti.subject_to(
            self.UB_eval.call([self.z])[0] - self.epsQ >= self.constraint
        )
        self.opti.subject_to(self.z <= self.x_max)
        self.opti.subject_to(self.z >= self.x_min)
        self.opti.subject_to(self.LB_eval.call([self.z])[0] <= self.constraint)
        self.opti.subject_to(self.x[0] == self.p_start)
        self.opti.subject_to(self.x[self.H] == self.p_end)
        # % And choose a concrete value for p, latter can be passed from another location
        # self.opti.set_value(p, self.p)
        self.opti.minimize(
            10 * ca.sumsqr(self.z - self.x_goal_p) + 0.0001 * ca.sumsqr(self.u)
        )

    def GPformulation(self, x_goal):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(2, self.H + 1)
        self.u = self.opti.variable(2, self.H)
        self.z = self.opti.variable(2, 1)

        p = self.opti.parameter(2, 1)
        self.opti.set_initial(self.z, [0.3, -2])
        self.opti.set_initial(self.x[:, 0], self.p)
        for k in range(0, self.H):
            self.opti.set_initial(self.x[:, k + 1], self.p)
            self.opti.subject_to(self.x[:, k + 1] == self.x[:, k] + self.u[:, k])
            # self.opti.subject_to(lb(self.x[k+1]) >= 2.25)
            self.opti.subject_to(self.u[:, k] <= self.u_max)
            self.opti.subject_to(self.u[:, k] >= self.u_min)
            self.opti.subject_to(self.x[:, k + 1] <= self.x_max)
            self.opti.subject_to(self.x[:, k + 1] >= self.x_min)
            # self.opti.subject_to(self.LB_eval.call(
            #     [self.x[:, k+1]])[0] >= self.constraint)
            self.opti.subject_to(self.x[1, k + 1] == -2.0)
        # self.opti.subject_to(self.UB_eval.call(
        #     [self.x[:, self.Hm+1]])[0] - self.LB_eval.call([self.x[:, self.Hm+1]])[0] >= 0.01)
        # self.opti.subject_to(self.UB_eval.call(
        #     [self.x[0, self.Hm+1]])[0] - self.Lc*ca.sqrt(ca.sumsqr(self.x[0, self.Hm+1]-self.z[0, :])) >= self.constraint)

        self.opti.subject_to(self.UB_eval.call([self.z[0, :]])[0] >= 0.01)
        # self.opti.subject_to(self.LB_eval.call([self.z])[0] <= self.constraint)
        self.opti.subject_to(self.x[:, 0] == p)
        self.opti.subject_to(self.x[:, self.H] == p)
        # % And choose a concrete value for p, latter can be passed from another location
        self.opti.set_value(p, self.p)
        self.opti.minimize(
            ca.sumsqr(self.z - x_goal.numpy()) + 0.0001 * ca.sumsqr(self.u)
        )

    def solve(self):  # parameterize and then solve
        # https://or.stackexchange.com/questions/2669/ipopt-with-hsl-vs-mumps
        # for feasibility: https://coin-or.github.io/Ipopt/OPTIONS.html
        # p_opts = {"expand": True}
        # self.opti.print_options()
        s_opts = {
            "ipopt.max_iter": 15,
            "ipopt.tol": 1e-8,
            "ipopt.derivative_test": "none",
            "ipopt.linear_solver": self.optim_param["linear_solver"],
            "ipopt.inf_pr_output": "original",
        }
        self.opti.solver("ipopt", s_opts)
        # self.opti.solver("sqpmethod")
        try:
            sol = self.opti.solve()
        except:
            print("did not converge, let see .debug value")
            self.opti = self.opti.debug
        a = 1

    def print(self):
        print(self.opti)


class PytorchGPEvaluator(ca.Callback):
    def __init__(self, t_out, func, get_sparsity_out, opts={}):
        """
        t_in: list of inputs (pytorch tensors)
        t_out: list of outputs (pytorch tensors)
        """
        ca.casadi.Callback.__init__(self)
        self.t_out = t_out  # it is a function
        self.func = func
        self.get_sparsity_out = get_sparsity_out
        self.construct("PytorchGPEvaluator", opts)
        self.refs = []

    def update_func(self, function):
        self.func = function
        self.t_out = function  # it is a function

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(1, 2)

    def eval(self, x):
        print("in eval", x)
        print(
            "I am returning",
            [
                self.t_out(torch.from_numpy(x[0].toarray()).reshape(-1, 2))
                .detach()
                .numpy()
            ],
        )
        # return [self.t_out(x).detach().numpy()]
        return [
            self.t_out(torch.from_numpy(x[0].toarray()).reshape(-1, 2)).detach().numpy()
        ]

    def gradjac(self, x):
        jacob = torch.autograd.functional.jacobian(self.func, x)
        return jacob.reshape(1, 2).detach()

    def gradhes(self, x):
        jacob = torch.autograd.functional.hessian(self.func, x)
        return jacob.reshape(-1, 2).detach()

    def jacob_sparsity_out(self, i):
        return ca.Sparsity.dense(1, 2)

    def hessian_sparsity_out(self, i):
        return ca.Sparsity.dense(2, 2)

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        callback = PytorchGPEvaluator(self.gradjac, self.func, self.jacob_sparsity_out)
        callback.jacob_sparsity_out = self.hessian_sparsity_out
        callback.gradjac = self.gradhes

        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out form that Cas.casadiexpects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        print(
            name,
            nominal_in + nominal_out,
            callback.call(nominal_in),
            inames,
            onames,
            opts,
        )
        return ca.Function(
            name, nominal_in + nominal_out, callback.call(nominal_in), inames, onames
        )


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
