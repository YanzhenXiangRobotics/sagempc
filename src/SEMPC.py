# This is the algorithm file. It will be responsible to call environement,
# collect measurement, setup of MPC problem, call model, solver, etc.
import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
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
from src.agent import get_idx_from_grid

from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Int32

from mlsocket import MLSocket

HOST = "127.0.0.1"
PORT = 65432

from src.utils.mpc_ref_tracker_node import MPCRefTracker
from src.agent import dynamics


class SEMPC(Node):
    def __init__(self, params, env, visu) -> None:
        super().__init__("sempc")
        # self.oracle_solver = Oracle_solver(params)
        self.use_isaac_sim = params["experiment"]["use_isaac_sim"]
        self.env = env
        self.fig_dir = os.path.join(self.env.env_dir, "figs")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.ref_path_publisher = self.create_publisher(
            Float64MultiArray, "/ref_path", 10
        )
        # self.complete_subscriber = self.create_subscription(
        #     Int32, "/complete", self.clock_listener_callback, 10
        # )
        self.sempc_solver = SEMPC_solver(
            params,
            env.VisuGrid,
            env.ax,
            env.legend_handles,
            env.fig,
            visu,
            self.fig_dir,
            self.publisher,
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
            self.state_dim = self.n_order * self.x_dim + 1
        elif params["agent"]["dynamics"] == "nova_carter":
            self.state_dim = self.n_order * self.x_dim + 1
        else:
            self.state_dim = self.n_orer * self.x_dim
        self.obtained_init_state = False
        self.sempc_initialization()
        self.sim_iter = 0
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.has_legend = False
        self.ref_tracker = MPCRefTracker()

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

    def set_next_goal(self):
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
                torch.from_numpy(self.players[self.pl_idx].current_location),
                self.players[self.pl_idx].grid_V,
            )
            # self.players[self.pl_idx].update_pessimistic_graph(V_lower_Cx, init_node, self.q_th, Lc=0)
            # curr_node = self.players[self.pl_idx].get_nearest_pessi_idx(torch.from_numpy(self.players[self.pl_idx].current_location))
            # intersect_pessi_opti =  torch.max(V_upper_Cx-self.eps, V_lower_Cx+0.04)
            if self.params["agent"]["dynamics"] == "nova_carter":
                # offset = self.params["common"]["constraint"] - 0.4
                offset = -0.04
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
            curr_node = self.players[self.pl_idx].get_nearest_opti_idx(
                torch.from_numpy(self.players[self.pl_idx].current_location)
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
            self.env.legend_handles += [tmp_0, tmp_1, tmp_2]

        if self.params["visu"]["show"]:
            self.visu.UpdateIter(self.iter, -1)
            self.visu.UpdateObjectiveVisu(0, self.players, self.env, 0)
            self.visu.writer_gp.grab_frame()
            self.visu.writer_dyn.grab_frame()
            self.visu.f_handle["dyn"].savefig("temp1D.png")
            self.visu.f_handle["gp"].savefig("temp in prog2.png")
        print(bcolors.green + "Goal:", xi_star, " uncertainity:", w, bcolors.ENDC)
        return w

    def get_safe_init(self):
        init_xy = {}
        init_xy["Cx_X"] = [torch.from_numpy(self.x_curr[: self.x_dim])]
        init_xy["Fx_X"] = init_xy["Cx_X"].copy()
        init_xy["Cx_Y"] = torch.atleast_2d(torch.tensor(self.query_meas[0]))
        init_xy["Fx_Y"] = torch.atleast_2d(
            torch.tensor(
                [np.linalg.norm(self.x_curr[:-1] - np.array(self.x_goal), ord=2)]
            )
        )
        return init_xy

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

    def sempc_main(self):
        """_summary_ Responsible for initialization, logic for when to collect sample vs explore"""
        # if self.params["algo"]["strategy"] == "SEpessi" or "SEopti" or "goose":
        w = 100
        # while not self.players[self.pl_idx].infeasible:
        running_condition_true = True
        self.players[self.pl_idx].feasible = True
        while running_condition_true:
            self.not_reached_and_prob_feasible()

            if w < self.params["common"]["epsilon"]:
                self.players[self.pl_idx].feasible = False
            else:
                ckp = time.time()
                w = self.set_next_goal()
                print(f"Time for finding next goal: {time.time() - ckp}")

            if self.params["algo"]["objective"] == "GO":
                running_condition_true = (
                    not np.linalg.norm(
                        self.visu.utility_minimizer
                        - self.players[self.pl_idx].current_location
                    )
                    < self.params["visu"]["step_size"]
                )
            elif self.params["algo"]["objective"] == "SE":
                running_condition_true = self.players[self.pl_idx].feasible
            else:
                raise NameError("Objective is not clear")
        print("Number of samples", self.players[self.pl_idx].Cx_X_train.shape)

    def sempc_initialization(self):
        if self.use_isaac_sim:
            while not self.obtained_init_state:
                self.get_current_state_measurement()
                print("waiting...")
            print("initialized location", self.get_safe_init())
        else:
            print("initialized location", self.env.get_safe_init())
        """_summary_ Everything before the looping for gp-measurements"""
        # 1) Initialize players to safe location in the environment
        # TODO: Remove dependence of player on visu grid
        if self.use_isaac_sim:
            self.players = get_players_initialized_isaac_sim(
                self.get_safe_init(),
                torch.tensor(self.x_curr)[: self.x_dim],
                self.params,
                self.env.VisuGrid,
            )
        else:
            self.players = get_players_initialized(
                self.env.get_safe_init(), self.params, self.env.VisuGrid
            )

        for it, player in enumerate(self.players):
            player.update_Cx_gp_with_current_data()
            init = self.env.get_safe_init()["Cx_X"][it].reshape(-1, 2).numpy()
            state = np.zeros(self.state_dim + 1)
            state[: self.x_dim] = init
            state[self.x_dim] = self.params["env"]["start_angle"]
            if self.use_isaac_sim:
                self.get_current_state_measurement()
            else:
                player.update_current_state(state)

        associate_dict = {}
        associate_dict[0] = []
        for idx in range(self.params["env"]["n_players"]):
            associate_dict[0].append(idx)

        # 2) Set goal based on strategy
        self.set_next_goal()

        # initial measurement (make sure l(x_init) >= 0)
        val = -100
        while val <= self.q_th:
            if self.use_isaac_sim:
                self.get_current_state_measurement()
                if self.params["experiment"]["batch_update"]:
                    query_pts = self.query_pts
                    query_meas = self.query_meas
                else:
                    query_pts = self.query_pts[0]
                    query_meas = self.query_meas[0]
                TrainAndUpdateConstraint_isaac_sim(
                    query_pts,
                    query_meas,
                    self.pl_idx,
                    self.players,
                    self.params,
                )
            else:
                TrainAndUpdateConstraint(
                    self.players[self.pl_idx].current_location,
                    self.pl_idx,
                    self.players,
                    self.params,
                    self.env,
                )
            val = self.players[self.pl_idx].get_lb_at_curr_loc()

        self.max_density_sigma = sum(
            [player.max_density_sigma for player in self.players]
        )
        self.data["sum_max_density_sigma"] = []
        self.data["sum_max_density_sigma"].append(self.max_density_sigma)
        print(self.iter, self.max_density_sigma)

    def get_current_state_measurement(self):
        try:
            s = MLSocket()
            s.connect((HOST, PORT))
            data = s.recv(1024)
            s.close()

            self.x_curr = data[: self.state_dim]
            min_dist_angle = data[self.state_dim]
            min_dist = data[self.state_dim + 1]
            self.t_curr = data[-1]

            query_pts_x_start = self.x_curr[0]
            query_pts_x_end = self.x_curr[0] + min_dist * np.cos(min_dist_angle)
            # resolution = 0.3 if query_pts_x_start <= query_pts_x_end else -0.3
            num_pts = self.params["experiment"]["batch_size"]
            query_pts_x = np.linspace(
                query_pts_x_start,
                query_pts_x_end,
                num_pts,
            )
            query_pts_y = np.linspace(
                self.x_curr[1],
                self.x_curr[1]
                + (query_pts_x[-1] - query_pts_x[0]) * np.tan(min_dist_angle),
                len(query_pts_x),
            )
            self.query_pts = np.vstack((query_pts_x, query_pts_y)).T
            self.query_meas = np.linspace(
                min_dist,
                min_dist - (query_pts_x[-1] - query_pts_x[0]) / np.cos(min_dist_angle),
                len(query_pts_x),
            )
            self.obtained_init_state = True

            self.players[self.pl_idx].update_current_state(self.x_curr)

        except Exception as e:
            print(e)

    def _angle_helper(self, angle):
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def apply_control_once(self, u):
        msg = Twist()
        self.get_current_state_measurement()
        start = self.t_curr
        while self.t_curr - start < u[-1]:
            msg.linear.x = u[0]
            msg.angular.z = u[1]
            self.publisher.publish(msg)
            self.get_current_state_measurement()

    def apply_control(self, path, ctrl, duration):
        msg = Twist()
        # for i in range(U.shape[0] - 1):
        for i in range(self.Hm):
            self.get_current_state_measurement()
            start = self.t_curr
            while self.t_curr - start < duration[i]:
                msg.linear.x = ctrl[i, 0]
                msg.angular.z = ctrl[i, 1]
                self.publisher.publish(msg)
                self.get_current_state_measurement()
                # print(
                #     f"Starting from {start} until {start + U[i, 2]} at {self.t_curr}, applied {U[i, :self.x_dim]}"
                # )
            uncertainty = self.players[self.pl_idx].get_width_at_curr_loc()
            print("uncertainty: ", uncertainty)
            # if uncertainty > 2.0 * self.params["common"]["epsilon"]:
            #     break
        msg = Twist()
        self.publisher.publish(msg)

    def one_step_planner(self):
        """_summary_: Plans going and coming back all in one trajectory plan
        Input: current location, end location, dyn, etc.
        Process: Solve the NLP and simulate the system until the measurement collection point
        Output: trajectory
        """
        # questions:

        self.visu.UpdateIter(self.iter, -1)
        print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)

        # Write in MPC style to reach the goal. The main loop is outside
        if self.use_isaac_sim:
            x_curr = self.x_curr
        else:
            x_curr = (
                self.players[self.pl_idx]
                .current_state[: self.state_dim]
                .reshape(self.state_dim)
            )  # 3D
        self.sempc_solver.update_x_curr(x_curr)
        x_origin = self.players[
            self.pl_idx
        ].origin.numpy()  # origin: related to X_train, thus 2-dims
        if torch.is_tensor(x_curr):
            x_curr = x_curr.numpy()
        st_curr = np.zeros(self.state_dim + self.x_dim + 1)  # 4
        st_curr[: self.state_dim] = np.ones(self.state_dim) * x_curr
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
                st_lb = np.zeros(self.state_dim + 1)
                st_ub = np.zeros(self.state_dim + 1)
                st_lb[: self.state_dim] = np.array(self.params["optimizer"]["x_min"])
                st_ub[: self.state_dim] = np.array(self.params["optimizer"]["x_max"])
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
                st_origin = np.zeros(self.state_dim + 1)
            st_origin[: self.x_dim] = np.ones(self.x_dim) * x_origin
            st_origin[-1] = 1.0
            self.sempc_solver.ocp_solver.set(self.H, "lbx", st_origin)
            self.sempc_solver.ocp_solver.set(self.H, "ubx", st_origin)
            # if self.params["algo"]["type"] == "MPC_expander" or self.params["algo"]["type"] == "ret_expander":
            #     pt_in_exp_lb = np.ones(self.state_dim+1)*(-1e8)
            #     pt_in_exp_ub = np.ones(self.state_dim+1)*(1e8)
            #     # pt_in_exp_lb[:self.x_dim] = self.players[self.pl_idx].get_next_to_go_loc() - 0.01
            #     # pt_in_exp_ub[:self.x_dim] = self.players[self.pl_idx].get_next_to_go_loc() + 0.01
            #     self.sempc_solver.ocp_solver.set(self.Hm, "lbx", pt_in_exp_lb)
            #     self.sempc_solver.ocp_solver.set(self.Hm, "ubx", pt_in_exp_ub)

        # set objective as per desired goal
        start_time = time.time()
        X, U = self.sempc_solver.solve(self.players[self.pl_idx], self.sim_iter)
        end_time = time.time()
        self.visu.time_record(end_time - start_time)
        # X, U, Sl = self.sempc_solver.get_solution()
        if self.use_isaac_sim:
            # self.apply_control(
            #     X[: self.Hm, : self.x_dim], X[: self.Hm, 3:5], U[: self.Hm, 2]
            # )
            self.inner_loop_control(X, x_curr)
        val = (
            2
            * self.players[self.pl_idx].Cx_beta
            * 2
            * torch.sqrt(
                self.players[self.pl_idx]
                .Cx_model(
                    torch.from_numpy(X[self.Hm, : self.x_dim]).reshape(-1, 2).float()
                )
                .variance
            )
            .detach()
            .item()
        )
        # print("slack", Sl, "uncertainity", X[self.Hm], val)#, "z-x",np.linalg.norm(X[:-1,0:2] - U[:,3:5]))
        # self.visu.record(X, U, X[self.Hm], self.pl_idx, self.players)
        self.visu.record(
            X,
            U,
            self.players[self.pl_idx].get_next_to_go_loc(),
            self.pl_idx,
            self.players,
        )

        # Environement simulation
        # x_curr = X[0]
        # for i in range(self.Hm):
        #     self.env.integrator.set("x", x_curr)
        #     self.env.integrator.set("u", U[i])
        #     self.env.integrator.solve()
        #     x_curr = self.env.integrator.get("x")
        #     if self.x_dim == 1:
        #         x_curr = np.hstack([x_curr[:self.x_dim].item(), -2.0])
        #     self.players[self.pl_idx].update_current_state(x_curr)
        self.players[self.pl_idx].safe_meas_loc = X[self.Hm][: self.x_dim]
        if (
            self.params["algo"]["type"] == "ret"
            or self.params["algo"]["type"] == "ret_expander"
        ):
            self.players[self.pl_idx].update_current_state(X[self.H])
            if self.goal_in_pessi:
                # if np.linalg.norm(self.visu.utility_minimizer-self.players[self.pl_idx].safe_meas_loc) < 0.025:
                self.players[self.pl_idx].update_current_state(X[self.Hm])
        else:
            if self.use_isaac_sim:
                self.get_current_state_measurement()
            else:
                print(
                    f"X first half: {X[:self.Hm, :]}, \n U first half: {U[:self.Hm, :3]}"
                )
                # self.players[self.pl_idx].update_current_state(X[self.Hm])
                self.inner_loop_control(X, x_curr)
                # ref_path_msg = Float64MultiArray()
                # ref_path_msg.data = (
                #     np.concatenate(
                #         (
                #             X[: self.Hm + 1, : self.x_dim],
                #             X[: self.Hm + 1, self.state_dim].reshape(-1, 1),
                #         ),
                #         axis=-1,
                #     )
                #     .flatten()
                #     .tolist()
                # )
                # self.ref_path_publisher.publish(ref_path_msg)
                # self.players[self.pl_idx].rollout(U[: self.Hm, :])
                x_curr = (
                    self.players[self.pl_idx]
                    .current_state[: self.state_dim]
                    .reshape(self.state_dim)
                )

        # assert np.isclose(x_curr,X[self.Hm]).all()
        # self.visu.UpdateIter(self.iter+i, -1)
        # self.visu.UpdateSafeVisu(0, self.players, self.env)
        # self.visu.writer_gp.grab_frame()
        # self.visu.writer_dyn.grab_frame()
        # self.visu.f_handle["dyn"].savefig("temp1D.png")
        # self.visu.f_handle["gp"].savefig(
        #     str(self.iter) + 'temp in prog2.png')
        if self.use_isaac_sim:
            self.env.legend_handles.append(
                self.env.ax.scatter(
                    self.x_curr[0],
                    self.x_curr[1],
                    color="red",
                    s=6,
                    label="actual trajectory",
                )
            )
        else:
            print(f"Red dot loc: {x_curr}")
            self.env.legend_handles.append(
                self.env.ax.scatter(
                    x_curr[0], x_curr[1], color="red", s=6, label="actual trajectory"
                )
            )
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
                    self.params["env"]["start"][0],
                    self.params["env"]["goal_loc"][0] + 3.0,
                ]
            )
            self.env.ax.set_ylim(
                [
                    self.params["env"]["start"][1],
                    self.params["env"]["goal_loc"][1] + 0.5,
                ]
            )
            self.env.ax.grid()
        self.env.fig.savefig(os.path.join(self.fig_dir, f"sim_{self.sim_iter}.png"))
        # self.env.fig.savefig(os.path.join(self.fig_dir, "sim.png"))
        self.sempc_solver.fig_3D.savefig(os.path.join(self.fig_dir, "sim_3D.png"))
        len_plot_tmps = len(self.sempc_solver.plot_tmps)
        len_scatter_tmps = len(self.sempc_solver.scatter_tmps)
        len_threeD_tmps = len(self.sempc_solver.threeD_tmps)
        for _ in range(len_plot_tmps):
            self.sempc_solver.plot_tmps.pop(0).remove()
        for _ in range(len_scatter_tmps):
            self.sempc_solver.scatter_tmps.pop(0).set_visible(False)
        for _ in range(len_threeD_tmps):
            self.sempc_solver.threeD_tmps.pop(0).remove()
        if self.use_isaac_sim:
            x_curr = self.x_curr
        print(
            bcolors.green + "Reached:",
            x_curr,
            X[self.Hm, : self.state_dim],
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
        self.sim_iter += 1

    def inner_loop_control(self, X, x_curr):
        self.ref_tracker.set_ref_path(X.tolist())
        # U_cl = np.zeros((self.Hm, self.x_dim))

        sagempc_sol_plot = self.env.ax.plot(
            X[: self.Hm, 0], X[: self.Hm, 1], c="black", marker="x", markersize=5
        )
        for k in range(self.Hm):
            x0 = self.players[self.pl_idx].state_sim[:-1].copy()
            # print("Pos 1: ", self.players[self.pl_idx].state_sim)
            X_inner, U_inner = self.ref_tracker.solve_for_x0(x0)
            if k == 0:
                print("Ol-Cl diff: ", X_inner - X)
            print(
                f"Sim iter: {self.sim_iter}, Stage: {k}, X inner: {X_inner}, U inner: {U_inner}"
            )
            if self.use_isaac_sim:
                self.apply_control_once(U_inner[0, :])
            else:
                self.players[self.pl_idx].rollout(U_inner[0, :].reshape(1, -1))
            # self.players[self.pl_idx].rollout(U[k, : self.x_dim + 1].reshape(1, -1))
            # print("Err 1: ", X[k + 1, :] - dynamics(X[k, :], U[k, : self.x_dim + 1]))
            # print("Err 2: ", X[k + 1, :] - self.players[self.pl_idx].state_sim)
            # print("Pos 2: ", self.players[self.pl_idx].state_sim)
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
        sagempc_sol_plot.pop(0).remove()

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
        # if self.flag_reached_xt_goal:
        #     self.visu.UpdateIter(self.iter, -1)
        #     self.visu.UpdateSafeVisu(0, self.players, self.env)
        #     self.visu.writer_gp.grab_frame()
        #     self.visu.writer_dyn.grab_frame()
        #     self.visu.f_handle["dyn"].savefig("temp1D.png")
        #     self.visu.f_handle["gp"].savefig('temp in prog2.png')
        #     return None
        # collect measurement at the current location
        # if problem is infeasible then also return
        if not self.goal_in_pessi:
            print(
                "Uncertainity at meas_loc",
                self.players[self.pl_idx].get_width_at_curr_loc(),
            )
            if self.use_isaac_sim:
                self.get_current_state_measurement()
                if self.params["experiment"]["batch_update"]:
                    query_pts = self.query_pts
                    query_meas = self.query_meas
                else:
                    query_pts = self.query_pts[0]
                    query_meas = self.query_meas[0]
                TrainAndUpdateConstraint_isaac_sim(
                    query_pts,
                    query_meas,
                    self.pl_idx,
                    self.players,
                    self.params,
                )
            else:
                ckp = time.time()
                TrainAndUpdateConstraint(
                    self.players[self.pl_idx].safe_meas_loc,
                    self.pl_idx,
                    self.players,
                    self.params,
                    self.env,
                )
                print(f"Time for GP update: {time.time() - ckp}")
            print(
                "Uncertainity at meas_loc",
                self.players[self.pl_idx].get_width_at_curr_loc(),
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
