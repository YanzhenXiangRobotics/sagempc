# This is the algorithm file. It will be responsible to call environement,
# collect measurement, setup of MPC problem, call model, solver, etc.
import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)
import torch
import time
from src.solver import Oracle_solver, SEMPC_solver
from src.utils.helper import (
    TrainAndUpdateConstraint,
    TrainAndUpdateConstraint_isaac_sim,
)
from src.utils.initializer import (
    get_players_initialized,
    get_players_initialized_isaac_sim,
)
from src.utils.termcolor import bcolors

import math
from src.agent import get_idx_from_grid, plot_graph_nodes

from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Int32

from mlsocket import MLSocket

HOST = "127.0.0.1"
PORT = 65432

from src.utils.inner_control_node import InnerControl
from src.agent import dynamics

import shutil


class SEMPC:
    def __init__(self, params, env, visu, query_state_obs_noise=None) -> None:
        # self.oracle_solver = Oracle_solver(params)
        self.use_isaac_sim = params["experiment"]["use_isaac_sim"]
        self.debug = params["experiment"]["debug"]
        self.env = env
        self.fig_dir = os.path.join(self.env.env_dir, "figs")
        # self.complete_subscriber = self.create_subscription(
        #     Int32, "/complete", self.clock_listener_callback, 10
        # )
        if os.path.exists(self.fig_dir):
            shutil.rmtree(self.fig_dir)
        self.sempc_solver = SEMPC_solver(
            params,
            env.VisuGrid,
            env.ax,
            env.legend_handles,
            env.fig,
            visu,
            self.fig_dir,
        )
        self.visu = visu
        self.params = params
        self.iter = -1
        self.data = {}
        self.flag_reached_xt_goal = False
        self.flag_new_goal = True
        self.pl_idx = 0  # currently single player, so does not matter
        self.H = self.params["optimizer"]["H"]
        self.Hm = self.params["optimizer"]["Hm"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        self.x_goal = params["env"]["goal_loc"]
        self.eps = params["common"]["epsilon"]
        self.q_th = params["common"]["constraint"]
        self.prev_goal_dist = 100
        self.goal_in_pessi = False
        if params["agent"]["dynamics"] == "robot":
            self.pose_dim = self.n_order * self.x_dim + 1
        elif params["agent"]["dynamics"] == "nova_carter":
            self.pose_dim = self.n_order * self.x_dim + 1
        else:
            self.pose_dim = self.n_order * self.x_dim
        self.obtained_init_state = False
        self.sempc_initialization(query_state_obs_noise)
        self.sim_iter = 0
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.has_legend = False
        self.ref_tracker = InnerControl()
        self.find_next_goal_time = []
        self.solve_time = []
        self.gp_update_time = []
        self.inner_loop_time = []
        self.one_iter_time = []

    def get_optimistic_path(self, node, goal_node, init_node):
        # If there doesn't exists a safe path then re-evaluate the goal
        try:
            print("init_node", init_node.item(), "goal_node", goal_node.item())
            t1 = self.players[self.pl_idx].get_optimistic_path(
                node.item(), goal_node.item()
            )
            t2 = self.players[self.pl_idx].get_optimistic_path(
                goal_node.item(), init_node.item()
            )
            opti_path = t1 + t2
        except:  # change of utility minimizer location
            list_opti_node = list(self.players[self.pl_idx].optimistic_graph.nodes())
            val_optimistic_graph = self.env.get_true_objective_func()[list_opti_node]
            self.players[self.pl_idx].get_utility_minimizer = (
                self.players[self.pl_idx]
                .grid_V[list_opti_node[val_optimistic_graph.argmin().item()]]
                .numpy()
            )
            goal_node = get_idx_from_grid(
                torch.from_numpy(self.players[self.pl_idx].get_utility_minimizer),
                self.players[self.pl_idx].grid_V,
            )
            self.visu.utility_minimizer = self.players[
                self.pl_idx
            ].get_utility_minimizer
            print(
                "init_node",
                init_node.item(),
                "curr_node",
                node.item(),
                "goal_node",
                goal_node.item(),
            )
            t1 = self.players[self.pl_idx].get_optimistic_path(
                node.item(), goal_node.item()
            )
            t2 = self.players[self.pl_idx].get_optimistic_path(
                goal_node.item(), init_node.item()
            )
            opti_path = t1 + t2
        return opti_path, goal_node

    def set_next_goal(self, x_curr=None):
        if x_curr is None:
            x_curr = self.players[self.pl_idx].current_location
        # 2) Set goal based on objective and strategy
        if self.params["algo"]["objective"] == "SE":
            if self.params["algo"]["strategy"] == "SEpessi":
                w, xi_star = self.players[self.pl_idx].uncertainity_sampling(
                    const_set="pessi"
                )
                self.players[self.pl_idx].set_maximizer_goal(xi_star)
            elif self.params["algo"]["strategy"] == "SEopti":
                w, xi_star = self.players[self.pl_idx].uncertainity_sampling(
                    const_set="opti"
                )
                self.players[self.pl_idx].set_maximizer_goal(xi_star)
            elif self.params["algo"]["strategy"] == "goose":
                # Set the x_g which can be used in the distance cost function
                xi_star = np.array(self.params["env"]["goal_loc"])
                self.players[self.pl_idx].set_maximizer_goal(xi_star)
                w = 100
            elif self.params["algo"]["strategy"] == "optiTraj":
                xi_star = self.oracle()
                self.players[self.pl_idx].set_maximizer_goal(xi_star)
                w = 100
            self.visu.num_safe_nodes = self.players[self.pl_idx].num_safe_nodes
            self.visu.opti_path = None
            self.visu.utility_minimizer = None

        elif self.params["algo"]["objective"] == "GO":
            V_lower_Cx, V_upper_Cx = self.players[self.pl_idx].get_Cx_bounds(
                self.players[self.pl_idx].grid_V
            )
            self.visu.num_safe_nodes = len(V_lower_Cx[V_lower_Cx > 0])
            init_node = get_idx_from_grid(
                self.players[self.pl_idx].origin, self.players[self.pl_idx].grid_V
            )
            curr_node = get_idx_from_grid(
                torch.from_numpy(x_curr),
                self.players[self.pl_idx].grid_V,
            )
            # self.players[self.pl_idx].update_pessimistic_graph(V_lower_Cx, init_node, self.q_th, Lc=0)
            # curr_node = self.players[self.pl_idx].get_nearest_pessi_idx(torch.from_numpy(self.players[self.pl_idx].current_location))
            # intersect_pessi_opti =  torch.max(V_upper_Cx-self.eps, V_lower_Cx+0.04)
            if self.params["agent"]["dynamics"] == "nova_carter":
                # offset = self.params["common"]["constraint"] - 0.4
                offset = -0.15
            elif self.params["experiment"]["folder"] == "cluttered_envs":
                offset = 0.05
            intersect_pessi_opti = V_upper_Cx - self.eps - offset
            X1, X2 = self.visu.x.numpy(), self.visu.y.numpy()
            intersect_pessi_opti_plot = (
                intersect_pessi_opti.detach().numpy().reshape(X1.shape[0], X2.shape[1])
            )
            if self.params["experiment"]["plot_contour"]:
                tmp_0 = self.env.ax.contour(
                    X1,
                    X2,
                    intersect_pessi_opti_plot,
                    levels=[self.params["common"]["constraint"]],
                    colors="green",
                    linewidths=0.5,
                )
                # tmp_0.collections[0].set_label("optimistic contour")
                (artists,), _ = tmp_0.legend_elements()
                artists.set_label(
                    "optimistic - eps(%.2f) - offset(%.2f) contour" % (self.eps, offset)
                )
                self.env.legend_handles.append(artists)
            self.players[self.pl_idx].update_optimistic_graph(
                intersect_pessi_opti, init_node, self.q_th, curr_node, Lc=0
            )
            # nodes = np.array(list(self.players[self.pl_idx].optimistic_graph.nodes))[::10]
            # plot_graph_nodes(
            #     self.env.ax,
            #     self.sempc_solver.scatter_tmps,
            #     nodes,
            #     self.players[self.pl_idx].grid_V
            # )
            curr_node = self.players[self.pl_idx].get_nearest_opti_idx(
                torch.from_numpy(x_curr)
            )
            self.players[self.pl_idx].get_utility_minimizer = np.array(
                self.params["env"]["goal_loc"]
            )
            goal_node = get_idx_from_grid(
                torch.from_numpy(self.players[self.pl_idx].get_utility_minimizer),
                self.players[self.pl_idx].grid_V,
            )
            self.visu.utility_minimizer = self.players[
                self.pl_idx
            ].get_utility_minimizer
            if (self.params["algo"]["type"] == "MPC_V0") or (
                self.params["algo"]["type"] == "MPC_expander_V0"
            ):
                opti_path, goal_node = self.get_optimistic_path(
                    curr_node, goal_node, init_node
                )
            else:
                opti_path, goal_node = self.get_optimistic_path(
                    init_node, goal_node, init_node
                )
            self.visu.opti_path = self.players[self.pl_idx].grid_V[opti_path]
            # if goal is within the pessimitic set set xi_star as goal directly
            if V_lower_Cx[goal_node] >= 0:
                xi_star = self.players[self.pl_idx].grid_V[goal_node.item()].numpy()
                self.goal_in_pessi = True
                self.players[self.pl_idx].goal_in_pessi = True
            else:
                pessi_value = V_lower_Cx[opti_path]
                if (
                    self.params["algo"]["type"] == "ret_expander"
                    or self.params["algo"]["type"] == "MPC_expander"
                    or self.params["algo"]["type"] == "MPC_expander_V0"
                ):
                    pessi_value = self.enlarge_pessi_set(pessi_value)
                idx_out_pessi = np.where(pessi_value < self.q_th)[0][0].item()
                xi_star = (
                    self.players[self.pl_idx].grid_V[opti_path[idx_out_pessi]].numpy()
                )
            self.players[self.pl_idx].set_maximizer_goal(xi_star)
            w = 100

            (tmp_1,) = self.env.ax.plot(
                self.visu.opti_path[:, 0],
                self.visu.opti_path[:, 1],
                c="violet",
                linewidth=0.5,
                label="A* path",
            )
            tmp_2 = self.env.ax.scatter(
                xi_star[0], xi_star[1], marker="x", s=30, c="violet", label="next goal"
            )
            self.sempc_solver.threeD_tmps.append(tmp_0)
            self.sempc_solver.plot_tmps.append(tmp_1)
            self.sempc_solver.scatter_tmps.append(tmp_2)
            # self.env.legend_handles += [tmp_0, tmp_1, tmp_2]

        if self.params["visu"]["show"]:
            self.visu.UpdateIter(self.iter, -1)
            self.visu.UpdateObjectiveVisu(0, self.players, self.env, 0)
            self.visu.writer_gp.grab_frame()
            self.visu.writer_dyn.grab_frame()
            self.visu.f_handle["dyn"].savefig("temp1D.png")
            self.visu.f_handle["gp"].savefig("temp in prog2.png")
        print(bcolors.green + "Goal:", xi_star, " uncertainity:", w, bcolors.ENDC)
        return w

    def enlarge_pessi_set(self, pessi_value):
        # Make it more efficient by only using matrix instead of vectors
        max_idx = pessi_value.argmax().item()
        cur_max_val = pessi_value[max_idx]
        cur_max_idx = max_idx
        for i in range(0, len(pessi_value)):
            if pessi_value[i] > cur_max_val - self.params["common"]["Lc"] * self.params[
                "visu"
            ]["step_size"] * np.abs(i - cur_max_idx):
                cur_max_idx = i
                cur_max_val = pessi_value[i]
            else:
                pessi_value[i] = cur_max_val - self.params["common"][
                    "Lc"
                ] * self.params["visu"]["step_size"] * np.abs(i - cur_max_idx)
        return pessi_value

    def running_condition_true_go(self, x_curr=None):
        minimizer = self.visu.utility_minimizer
        current_location = (
            self.players[self.pl_idx].current_location if x_curr is None else x_curr
        )
        resolution = self.params["visu"]["step_size"]

        return np.linalg.norm(minimizer - current_location) >= resolution

    def sempc_main(self):
        """_summary_ Responsible for initialization, logic for when to collect sample vs explore"""
        # if self.params["algo"]["strategy"] == "SEpessi" or "SEopti" or "goose":
        w = 100
        # while not self.players[self.pl_idx].infeasible:
        running_condition_true = True
        self.players[self.pl_idx].feasible = True
        last_time = time.time()
        while running_condition_true:
            self.not_reached_and_prob_feasible()

            if w < self.params["common"]["epsilon"]:
                self.players[self.pl_idx].feasible = False
            else:
                ckp = time.time()
                w = self.set_next_goal()
                time_find_next_goal = time.time() - ckp
                self.find_next_goal_time.append(time_find_next_goal)
                find_next_goal_time = np.array(self.find_next_goal_time)
                print(
                    f"Time finding next goal: {time_find_next_goal}, mean: {np.mean(find_next_goal_time)}, std: {np.std(find_next_goal_time)}"
                )

            if self.params["algo"]["objective"] == "GO":
                running_condition_true = self.running_condition_true_go()
            elif self.params["algo"]["objective"] == "SE":
                running_condition_true = self.players[self.pl_idx].feasible
            else:
                raise NameError("Objective is not clear")
            time_one_iter = time.time() - last_time
            last_time = time.time()
            if self.sim_iter > 1:
                self.one_iter_time.append(time_one_iter)
                one_iter_time = np.array(self.one_iter_time)
                print(
                    f"One iter time: {time_one_iter}, mean: {np.mean(one_iter_time)}, std: {np.std(one_iter_time)}\n"
                )
        print("Number of samples", self.players[self.pl_idx].Cx_X_train.shape)

    def sempc_initialization(self, query_state_obs_noise=None):
        if query_state_obs_noise is None:
            init_loc_obs_noise = self.env.get_safe_init()
        else:
            query_state_obs_noise_tensor = torch.from_numpy(query_state_obs_noise)
            init_loc_obs_noise = {}
            init_loc_obs_noise["Cx_X"] = [query_state_obs_noise_tensor[: self.x_dim]]
            init_loc_obs_noise["Cx_Y"] = torch.atleast_2d(
                query_state_obs_noise_tensor[self.x_dim]
            )
            init_loc_obs_noise["Cx_noise"] = torch.atleast_2d(
                query_state_obs_noise_tensor[-1]
            )
        print("initialized location observation", init_loc_obs_noise)
        """_summary_ Everything before the looping for gp-measurements"""
        # 1) Initialize players to safe location in the environment
        # TODO: Remove dependence of player on visu grid
        self.players = get_players_initialized(
            init_loc_obs_noise, self.params, self.env.VisuGrid
        )

        self.players[0].update_Cx_gp_with_current_data()
        if query_state_obs_noise_tensor is None:
            x_curr = init_loc_obs_noise["Cx_X"][0].numpy()
            pose_curr = np.append(x_curr, self.params["env"]["start_angle"])
        else:
            pose_curr = query_state_obs_noise_tensor[: self.x_dim + 1]
        self.players[0].update_current_state(pose_curr)

        associate_dict = {}
        associate_dict[0] = []
        for idx in range(self.params["env"]["n_players"]):
            associate_dict[0].append(idx)

        # 2) Set goal based on strategy
        self.set_next_goal(
            query_state_obs_noise[: self.x_dim]
            if query_state_obs_noise is not None
            else None
        )

        # initial measurement (make sure l(x_init) >= 0)
        val = -100
        while val <= self.q_th:
            if query_state_obs_noise_tensor is None:
                TrainAndUpdateConstraint(
                    self.players[self.pl_idx].current_location,
                    self.pl_idx,
                    self.players,
                    self.params,
                    self.env,
                )
            else:
                TrainAndUpdateConstraint_isaac_sim(
                    query_state_obs_noise[: self.x_dim],
                    query_state_obs_noise[self.x_dim],
                    # query_state_obs_noise[-1],
                    0.0001,
                    self.pl_idx,
                    self.players,
                    self.params,
                )
            val = self.players[self.pl_idx].get_lb_at_curr_loc()

        self.max_density_sigma = sum(
            [player.max_density_sigma for player in self.players]
        )
        self.data["sum_max_density_sigma"] = []
        self.data["sum_max_density_sigma"].append(self.max_density_sigma)
        print(self.iter, self.max_density_sigma)

    def prepare(self, state_curr=None):
        # self.visu.UpdateIter(self.iter, -1)
        print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)

        # Write in MPC style to reach the goal. The main loop is outside
        if state_curr is None:
            state_curr = np.zeros(self.pose_dim + self.x_dim)
            state_curr[: self.pose_dim] = self.players[
                self.pl_idx
            ].current_state.reshape(self.pose_dim)  # 3D
        self.sempc_solver.update_x_curr(state_curr[: self.pose_dim])
        x_origin = self.players[
            self.pl_idx
        ].origin.numpy()  # origin: related to X_train, thus 2-dims
        if torch.is_tensor(state_curr):
            state_curr = state_curr.numpy()
        st_curr = np.zeros(self.pose_dim + self.x_dim + 1)  # 6
        st_curr[: self.pose_dim + self.x_dim] = (
            np.ones(self.pose_dim + self.x_dim) * state_curr
        )
        if self.debug:
            print(f"St curr: {st_curr}")
        self.sempc_solver.ocp_solver.set(0, "lbx", st_curr)
        self.sempc_solver.ocp_solver.set(0, "ubx", st_curr)
        if self.params["algo"]["type"] == "MPC_V0" or (
            self.params["algo"]["type"] == "MPC_expander_V0"
        ):
            if self.params["agent"]["dynamics"] == "nova_carter":
                st_lb = np.zeros(2 * self.x_dim + 1)
                st_ub = np.zeros(2 * self.x_dim + 1)
                st_lb[: self.x_dim] = np.array(self.params["optimizer"]["x_min"])[
                    : self.x_dim
                ]
                st_ub[: self.x_dim] = np.array(self.params["optimizer"]["x_max"])[
                    : self.x_dim
                ]
            else:
                st_lb = np.zeros(self.pose_dim + 1)
                st_ub = np.zeros(self.pose_dim + 1)
                st_lb[: self.pose_dim] = np.array(self.params["optimizer"]["x_min"])
                st_ub[: self.pose_dim] = np.array(self.params["optimizer"]["x_max"])
            if self.params["agent"]["dynamics"] == "robot":
                st_lb[3] = 0.0
                st_ub[3] = 0.0
                st_lb[4] = 0.0
                st_ub[4] = 0.0
            elif (
                self.params["agent"]["dynamics"] == "unicycle"
                or self.params["agent"]["dynamics"] == "bicycle"
            ):
                st_lb[3] = 0.0
                st_ub[3] = 0.0
            elif self.params["agent"]["dynamics"] == "int":
                st_lb[2] = 0.0
                st_ub[2] = 0.0
                st_lb[3] = 0.0
                st_ub[3] = 0.0
            # st_ub[2] = 6.28
            # st_lb[2] = -6.28
            st_ub[-1] = self.params["optimizer"]["Tf"]
            # self.sempc_solver.ocp_solver.set(self.Hm, "lbx", st_lb)
            # self.sempc_solver.ocp_solver.set(self.Hm, "ubx", st_ub)
            st_lb[-1] = self.params["optimizer"]["Tf"]
            self.sempc_solver.ocp_solver.set(self.H, "lbx", st_lb)
            self.sempc_solver.ocp_solver.set(self.H, "ubx", st_ub)
            lbx_m = st_lb.copy()
            lbx_m[2:] = np.array([0.0, 0.0, 0.0])
            ubx_m = st_ub.copy()
            ubx_m[2:4] = np.array([0.0, 0.0])
            self.sempc_solver.ocp_solver.set(self.Hm, "lbx", lbx_m)
            self.sempc_solver.ocp_solver.set(self.Hm, "ubx", ubx_m)
        else:
            if self.params["agent"]["dynamics"] == "nova_carter":
                st_origin = np.zeros(self.x_dim + 1)
            else:
                st_origin = np.zeros(self.pose_dim + 1)
            st_origin[: self.x_dim] = np.ones(self.x_dim) * x_origin
            st_origin[-1] = 1.0
            self.sempc_solver.ocp_solver.set(self.H, "lbx", st_origin)
            self.sempc_solver.ocp_solver.set(self.H, "ubx", st_origin)
            # if self.params["algo"]["type"] == "MPC_expander" or self.params["algo"]["type"] == "ret_expander":
            #     pt_in_exp_lb = np.ones(self.pose_dim+1)*(-1e8)
            #     pt_in_exp_ub = np.ones(self.pose_dim+1)*(1e8)
            #     # pt_in_exp_lb[:self.x_dim] = self.players[self.pl_idx].get_next_to_go_loc() - 0.01
            #     # pt_in_exp_ub[:self.x_dim] = self.players[self.pl_idx].get_next_to_go_loc() + 0.01
            #     self.sempc_solver.ocp_solver.set(self.Hm, "lbx", pt_in_exp_lb)
            #     self.sempc_solver.ocp_solver.set(self.Hm, "ubx", pt_in_exp_ub)
        return state_curr

    def one_step_planner_plot(self, X, x_curr):
        if self.debug:
            print(f"Red dot loc: {x_curr}")
        self.env.legend_handles.append(
            self.env.ax.scatter(
                x_curr[0], x_curr[1], color="red", s=6, label="actual trajectory"
            )
        )
        self.env.legend_handles.append(
            self.env.ax.plot(
                X[: self.Hm, 0],
                X[: self.Hm, 1],
                color="black",
                # linewidth=10,
            )
        )
        self.sempc_solver.plot_3D(self.players[self.pl_idx], self.sim_iter)
        # self.env.fig.savefig(os.path.join(self.fig_dir, f"sim_{self.sim_iter}.png"))
        # if not self.has_legend:
        #     # self.env.ax.legend(handles=self.env.legend_handles, loc="upper right")
        #     self.env.ax.legend(handles=self.env.legend_handles)
        #     self.has_legend = True
        if self.params["agent"]["dynamics"] == "nova_carter":
            # self.env.ax.set_xlim([-21.8, -9.0])
            # self.env.ax.set_ylim([-21.8, -4.0])

            self.env.ax.set_xlim(
                [
                    self.params["env"]["start"][0] + 0.5,
                    self.params["env"]["goal_loc"][0] + 0.5,
                ]
            )
            self.env.ax.set_ylim(
                [
                    self.params["env"]["start"][1] + 0.5,
                    self.params["env"]["goal_loc"][1] + 0.1,
                ]
            )
            # self.env.ax.grid()
        self.env.fig.savefig(os.path.join(self.fig_dir, f"sim_{self.sim_iter}.png"))
        # self.env.fig.savefig(os.path.join(self.fig_dir, "sim.png"))
        len_plot_tmps = len(self.sempc_solver.plot_tmps)
        len_scatter_tmps = len(self.sempc_solver.scatter_tmps)
        len_threeD_tmps = len(self.sempc_solver.threeD_tmps)
        for _ in range(len_plot_tmps):
            self.sempc_solver.plot_tmps.pop(0).remove()
        for _ in range(len_scatter_tmps):
            self.sempc_solver.scatter_tmps.pop(0).set_visible(False)
        for _ in range(len_threeD_tmps):
            self.sempc_solver.threeD_tmps.pop(0).remove()

    def one_step_planner(self, state_curr=None):
        """_summary_: Plans going and coming back all in one trajectory plan
        Input: current location, end location, dyn, etc.
        Process: Solve the NLP and simulate the system until the measurement collection point
        Output: trajectory
        """
        ckp = time.time()
        do_inner_loop = True if state_curr is None else False
        state_curr = self.prepare(state_curr)
        print(f"Time for prepare: {time.time() - ckp}")

        # set objective as per desired goal
        start_time = time.time()
        X, U = self.sempc_solver.solve(self.players[self.pl_idx], self.sim_iter)
        end_time = time.time()
        time_solve = end_time - start_time
        self.visu.time_record(time_solve)
        if self.sim_iter > 0:
            self.solve_time.append(time_solve)
            solve_time = np.array(self.solve_time)
            print(
                f"Time solving sagempc: {time_solve}, mean: {np.mean(solve_time)}, std: {np.std(solve_time)}"
            )
        self.players[self.pl_idx].safe_meas_loc = X[self.Hm][: self.x_dim]
        if (
            self.params["algo"]["type"] == "ret"
            or self.params["algo"]["type"] == "ret_expander"
        ):
            self.players[self.pl_idx].update_current_state(X[self.H])
            if self.goal_in_pessi:
                # if np.linalg.norm(self.visu.utility_minimizer-self.players[self.pl_idx].safe_meas_loc) < 0.025:
                self.players[self.pl_idx].update_current_state(X[self.Hm])
        elif do_inner_loop:
            ckp = time.time()
            # import pprint
            # pprint.pprint(X[: self.Hm, :])
            # self.players[self.pl_idx].update_current_state(X[self.Hm, :self.pose_dim])
            if self.params["experiment"]["use_isaac_sim"] == 1:
                X_cl, U_cl = self.inner_loop_mpc(X, state_curr[: self.pose_dim])
            elif self.params["experiment"]["use_isaac_sim"] == 2:
                X_cl, U_cl = self.inner_loop_nav2(X, state_curr[: self.pose_dim])
            X_cl[self.Hm :, :] = X[self.Hm :, :-1].copy()
            U_cl[self.Hm :, :] = U[self.Hm :, : self.x_dim].copy()
            X_cl = np.concatenate(
                (
                    X_cl,
                    np.linspace(
                        0.0, self.params["optimizer"]["Tf"], self.H + 1
                    ).reshape(-1, 1),
                ),
                axis=-1,
            )
            U_cl = np.concatenate(
                (
                    U_cl,
                    self.params["optimizer"]["Tf"] / self.H * np.ones((self.H, 1)),
                ),
                axis=-1,
            )
            time_inner_loop = time.time() - ckp
            self.inner_loop_time.append(time_inner_loop)
            inner_loop_time = np.array(self.inner_loop_time)
            print(
                f"Time for inner-loop control: {time.time() - ckp}, mean: {np.mean(inner_loop_time)} std: {np.std(inner_loop_time)}"
            )
            pose_curr = (
                self.players[self.pl_idx]
                .current_state[: self.pose_dim]
                .reshape(self.pose_dim)
            )
            # self.visu.record(
            #     X,
            #     U,
            #     self.players[self.pl_idx].get_next_to_go_loc(),
            #     self.pl_idx,
            #     self.players,
            # )
            self.visu.record(
                X_cl,
                U_cl,
                self.players[self.pl_idx].get_next_to_go_loc(),
                self.pl_idx,
                self.players,
            )
            # print(f"Time for plotting at each sim iter: {time.time() - ckp}")
            print(
                bcolors.green + "Reached:",
                pose_curr,
                X[self.Hm, : self.pose_dim],
                bcolors.ENDC,
            )
            # set current location as the location to be measured

            goal_dist = np.linalg.norm(
                self.players[self.pl_idx].planned_measure_loc
                - self.players[self.pl_idx].current_location
            )
            if np.abs(goal_dist) < 1.0e-2:
                self.flag_reached_xt_goal = True
        # if np.abs(self.prev_goal_dist - goal_dist) < 1e-2:
        #     # set infeasibility flag to true and ask for a new goal
        #     self.players[self.pl_idx].infeasible = True
        # self.prev_goal_dist = goal_dist
        # apply this input to your environment
        self.one_step_planner_plot(X, state_curr[: self.x_dim])
        self.sim_iter += 1

        return X, U

    def inner_loop_mpc(self, X, x_curr):
        self.ref_tracker.set_ref_path(X[:, : self.pose_dim].tolist())
        X_cl = np.zeros((self.H + 1, self.pose_dim + self.x_dim))
        U_cl = np.zeros((self.H, self.x_dim))
        X_cl[0, :] = self.players[self.pl_idx].state_sim[:-1].copy()
        if self.sempc_solver.debug:
            sagempc_sol_plot = self.env.ax.plot(
                X[: self.Hm, 0], X[: self.Hm, 1], c="black", marker="x", markersize=5
            )
        for k in range(self.Hm):
            x0 = self.players[self.pl_idx].state_sim[:-1].copy()
            # print("Pos 1: ", self.players[self.pl_idx].state_sim)
            before = time.time()
            X_inner, U_inner = self.ref_tracker.solve_for_x0(x0)
            # print(f"Time solving CL: {time.time() - before}")
            if self.debug:
                if k == 0:
                    print("Ol-Cl diff: ", X_inner - X[:, :-1])
                print(
                    f"Sim iter: {self.sim_iter}, Stage: {k}, X inner: {X_inner}, U inner: {U_inner}"
                )
            self.players[self.pl_idx].rollout(
                np.append(
                    U_inner[0, :],
                    self.params["optimizer"]["Tf"] / self.params["optimizer"]["H"],
                ).reshape(1, -1)
            )
            X_cl[k + 1, :] = X_inner[1, :]
            U_cl[k, :] = U_inner[0, :]
            if self.sempc_solver.debug:
                curr_loc_plot = self.env.ax.scatter(
                    x0[0],
                    x0[1],
                    c="green",
                    marker="*",
                    s=150,
                )
                inner_loop_plot = self.env.ax.plot(
                    X_inner[: self.Hm, 0],
                    X_inner[: self.Hm, 1],
                    c="blue",
                    marker="x",
                    markersize=5,
                )
                radius = (
                    self.sempc_solver.local_plot_radius * 3
                    if self.sim_iter < 10
                    else self.sempc_solver.local_plot_radius / 3
                )
                self.env.ax.set_xlim(
                    [
                        x_curr[0] - radius,
                        x_curr[0] + radius,
                    ]
                )
                self.env.ax.set_ylim(
                    [
                        x_curr[1] - radius,
                        x_curr[1] + radius,
                    ]
                )
                self.env.fig.savefig(
                    os.path.join("inner_loop", f"inner_loop_{self.sim_iter}_{k}.png")
                )
                inner_loop_plot.pop(0).remove()
                curr_loc_plot.set_visible(False)
        if self.sempc_solver.debug:
            sagempc_sol_plot.pop(0).remove()
        if self.use_isaac_sim:
            msg = Twist()
            self.publisher.publish(msg)
        return X_cl, U_cl

    def inner_loop_nav2(self, X, x_curr):
        pass

    def update_Cx_gp(self, query_loc_obs_noise=None):
        before = time.time()
        if not self.goal_in_pessi:
            print(
                "Uncertainity at meas_loc",
                self.players[self.pl_idx].get_width_at_curr_loc(),
            )
        if query_loc_obs_noise is None:
            TrainAndUpdateConstraint(
                self.players[self.pl_idx].safe_meas_loc,
                self.pl_idx,
                self.players,
                self.params,
                self.env,
            )
        else:
            TrainAndUpdateConstraint_isaac_sim(
                query_loc_obs_noise[:, : self.x_dim],
                query_loc_obs_noise[:, self.x_dim],
                query_loc_obs_noise[:, -1],
                self.pl_idx,
                self.players,
                self.params,
            )
            self.sempc_solver.scatter_tmps.append(
                self.env.ax.scatter(
                    query_loc_obs_noise[:, 0],
                    query_loc_obs_noise[:, 1],
                    marker="x",
                    color="purple",
                    s=6,
                )
            )
        print(
            "Uncertainity at meas_loc",
            self.players[self.pl_idx].get_width_at_curr_loc(),
        )
        print("Time updating GP: ", time.time() - before)

    def not_reached_and_prob_feasible(self):
        """_summary_ The agent safely explores and either reach goal or remove it from safe set
        (not reached_xt_goal) and (not player.infeasible)
        """
        # while not self.flag_reached_xt_goal and (not self.players[self.pl_idx].infeasible):
        # this while loops ensures we collect measurement only at constraint and not all along
        # the path
        # self.receding_horizon(self.players[self.pl_idx])
        ckp = time.time()
        self.one_step_planner()
        print(f"Time for one step planner: {time.time() - ckp}")

        ckp = time.time()
        self.update_Cx_gp()
        time_gp_update = time.time() - ckp
        self.gp_update_time.append(time_gp_update)
        gp_update_time = np.array(self.gp_update_time)
        print(
            f"Time for gp update: {time_gp_update}, mean: {np.mean(gp_update_time)}, std: {np.std(gp_update_time)}"
        )

        if self.params["visu"]["show"]:
            self.visu.UpdateIter(self.iter, -1)
            self.visu.UpdateSafeVisu(0, self.players, self.env)
            self.visu.writer_gp.grab_frame()
            self.visu.writer_dyn.grab_frame()
            # self.visu.f_handle["dyn"].savefig("temp1D.png")
            # self.visu.f_handle["gp"].savefig('temp in prog2.png')

    def goal_reached_or_prob_infeasible(self):
        self.iter += 1
        self.visu.UpdateIter(self.iter, -1)
        print(bcolors.OKGREEN + "Solving Objective" + bcolors.ENDC)
        # Fx_model = self.players[0].Fx_model.eval()

        # get new goal
        xi_star = self.oracle()
        new_goal = True
        dist = np.linalg.norm(
            self.players[self.pl_idx].planned_measure_loc
            - self.players[self.pl_idx].current_location
        )
        if np.abs(dist) > 1.0e-3:
            self.flag_reached_xt_goal = False
        self.players[self.pl_idx].infeasible = False
        self.visu.UpdateIter(self.iter, -1)
        self.visu.UpdateObjectiveVisu(0, self.players, self.env, 0)
        # self.visu.UpdateDynVisu(0, self.players)
        # reached_xt_goal = False
        # if LB(player.planned_measure_loc.reshape(-1, 2)) >= params["common"]["constraint"]:
        #     reached_xt_goal = True
        #     reached_pt = player.planned_measure_loc.reshape(-1, 2)
        #     player.update_current_location(reached_pt)
        # player.infeasible = False
        # pt_fx_dyn = self.visu.plot_Fx_traj(player, fig_dyn, pt_fx_dyn)

        self.visu.writer_gp.grab_frame()
        self.visu.writer_dyn.grab_frame()
        self.visu.f_handle["dyn"].savefig("temp1D.png")
        self.visu.f_handle["gp"].savefig("temp in prog2.png")
