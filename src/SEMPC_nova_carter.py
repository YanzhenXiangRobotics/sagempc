# This is the algorithm file. It will be responsible to call environement,
# collect measurement, setup of MPC problem, call model, solver, etc.
import os

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from src.solver import SEMPC_solver
from src.utils.helper import (
    TrainAndUpdateConstraint,
    TrainAndUpdateDensity,
    get_frame_writer,
    oracle,
)
from src.utils.initializer import get_players_initialized
from src.utils.termcolor import bcolors

from rclpy.node import Node
from geometry_msgs.msg import Twist

from mlsocket import MLSocket

HOST = "127.0.0.1"
PORT = 65432

plot = False

class SEMPCNovaCarter(Node):
    def __init__(self, params) -> None:
        super().__init__("sempc_nova_carter")
        # self.oracle_solver = Oracle_solver(params)
        self.sempc_solver = SEMPC_solver(params)
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
        self.u_dim = params["optimizer"]["u_dim"]
        self.eps = params["common"]["epsilon"]
        self.q_th = params["common"]["constraint"]
        self.x_goal = params["env"]["goal_loc"]
        self.prev_goal_dist = 100
        self.goal_in_pessi = False
        if params["agent"]["dynamics"] == "robot":
            self.state_dim = self.n_order * self.x_dim + 1
        else:
            self.state_dim = self.n_order * self.x_dim

        self.obtained_init_state = False
        self.sempc_initialization()

        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sample_iter = 0

    def sempc_main(self):
        """_summary_ Responsible for initialization, logic for when to collect sample vs explore"""
        # if self.params["algo"]["strategy"] == "SEpessi" or "SEopti" or "goose":
        w = 100
        # while not self.players[self.pl_idx].infeasible:
        running_condition_true = True
        self.players[self.pl_idx].feasible = True
        while running_condition_true:
            self.not_reached_and_prob_feasible()
            if self.sempc_solver.plotter.plot:
                self.sempc_solver.plotter.plot_sim(self.sample_iter, self.x_curr)

            if w < self.params["common"]["epsilon"]:
                self.players[self.pl_idx].feasible = False

            if self.params["algo"]["objective"] == "GO":
                running_condition_true = (
                    not np.linalg.norm(
                        self.x_goal
                        - self.players[self.pl_idx].current_location[: self.x_dim - 1]
                    )
                    < 0.025
                )
            elif self.params["algo"]["objective"] == "SE":
                running_condition_true = self.players[self.pl_idx].feasible
            else:
                raise NameError("Objective is not clear")

            self.sample_iter += 1

        print("Number of samples", self.players[self.pl_idx].Cx_X_train.shape)

    def get_safe_init(self):
        init_xy = {}
        init_xy["Cx_X"] = [torch.from_numpy(self.x_curr[:-1])]
        init_xy["Fx_X"] = init_xy["Cx_X"].copy()
        init_xy["Cx_Y"] = torch.atleast_2d(torch.tensor(self.min_dist))
        init_xy["Fx_Y"] = torch.atleast_2d(
            torch.tensor(
                [np.linalg.norm(self.x_curr[:-1] - np.array(self.x_goal), ord=2)]
            )
        )
        return init_xy

    def sempc_initialization(self):
        """_summary_ Everything before the looping for gp-measurements"""
        # 1) Initialize players to safe location in the environment
        while not self.obtained_init_state:
            self.get_current_state()
            print("waiting...")
        print("initialized location", self.get_safe_init())
        # TODO: Remove dependence of player on visu grid
        self.players = get_players_initialized(
            self.get_safe_init(), torch.tensor(self.x_curr), self.params
        )

        for it, player in enumerate(self.players):
            player.update_Cx_gp_with_current_data()
            player.update_Fx_gp_with_current_data()
            player.update_current_state(self.x_curr, self.t_curr)

        associate_dict = {}
        associate_dict[0] = []
        for idx in range(self.params["env"]["n_players"]):
            associate_dict[0].append(idx)

        # initial measurement (make sure l(x_init) >= 0)
        val = -100
        while val <= self.q_th:
            TrainAndUpdateConstraint(
                self.players[self.pl_idx].current_location[: self.x_dim - 1],
                self.min_dist,
                self.pl_idx,
                self.players,
                self.params,
            )
            val = self.players[self.pl_idx].get_lb_at_curr_loc()

        # 3) Set termination criteria
        self.max_density_sigma = sum(
            [player.max_density_sigma for player in self.players]
        )
        self.data["sum_max_density_sigma"] = []
        self.data["sum_max_density_sigma"].append(self.max_density_sigma)
        print(self.iter, self.max_density_sigma)

    def apply_control(self, U):
        print(U)
        for i in range(U.shape[0]):
            self.get_current_state()
            start = self.t_curr
            while self.t_curr - start < U[i, -1]:
                msg = Twist()
                msg.linear.x = U[i, 0]
                msg.linear.y = U[i, 1]
                msg.angular.z = U[i, 2]
                self.publisher.publish(msg)

                print(
                    f"Starting from {start} until {start + U[i, -1]} at {self.t_curr}, applied {U[i, :self.u_dim]}"
                )

                self.get_current_state()

    def get_current_state(self):
        try:
            s = MLSocket()
            s.connect((HOST, PORT))
            data = s.recv(1024)
            s.close()

            self.x_curr = data[: self.x_dim]
            self.min_dist = data[self.x_dim]
            self.t_curr = data[-1]

            self.obtained_init_state = True

        except Exception as e:
            print(e)

    def one_step_planner(self):
        """_summary_: Plans going and coming back all in one trajectory plan
        Input: current location, end location, dyn, etc.
        Process: Solve the NLP and simulate the system until the measurement collection point
        Output: trajectory
        """
        # questions:

        print(bcolors.OKCYAN + "Solving Constrints" + bcolors.ENDC)

        # Write in MPC style to reach the goal. The main loop is outside
        x_curr = (
            self.players[self.pl_idx]
            .current_state[: self.state_dim]
            .reshape(self.state_dim)
        )
        x_origin = self.players[self.pl_idx].origin[: self.x_dim].numpy()
        if torch.is_tensor(x_curr):
            x_curr = x_curr.numpy()
        st_curr = np.zeros(self.state_dim + 1)
        st_curr[: self.state_dim] = np.ones(self.state_dim) * x_curr
        # print(self.sempc_solver.ocp_solver.acados_ocp.constraints.lbx)
        # print(self.sempc_solver.ocp_solver.acados_ocp.constraints.ubx)
        # print(self.sempc_solver.ocp_solver.acados_ocp.constraints.idxbx)
        self.sempc_solver.ocp_solver.set(0, "lbx", st_curr)
        self.sempc_solver.ocp_solver.set(0, "ubx", st_curr)

        st_origin = np.zeros(self.state_dim + 1)
        st_origin[: self.x_dim] = np.ones(self.x_dim) * x_origin
        st_origin[-1] = 1.0
        self.sempc_solver.ocp_solver.set(self.H, "lbx", st_origin)
        self.sempc_solver.ocp_solver.set(self.H, "ubx", st_origin)

        # set objective as per desired goal
        start_time = time.time()
        self.sempc_solver.solve(self.players[self.pl_idx], self.sample_iter)
        end_time = time.time()
        X, U, Sl = self.sempc_solver.get_solution()
        self.apply_control(U)

        self.players[self.pl_idx].safe_meas_loc = X[self.Hm][: self.x_dim]

        self.get_current_state()
        self.players[self.pl_idx].update_current_state(self.x_curr, self.t_curr)

        print(bcolors.green + "Reached:", x_curr, bcolors.ENDC)
        # set current location as the location to be measured

        goal_dist = np.linalg.norm(
            self.players[self.pl_idx].planned_measure_loc
            - self.players[self.pl_idx].current_location
        )
        if np.abs(goal_dist) < 1.0e-2:
            self.flag_reached_xt_goal = True

    def not_reached_and_prob_feasible(self):
        """_summary_ The agent safely explores and either reach goal or remove it from safe set
        (not reached_xt_goal) and (not player.infeasible)
        """
        # while not self.flag_reached_xt_goal and (not self.players[self.pl_idx].infeasible):
        # this while loops ensures we collect measurement only at constraint and not all along
        # the path
        # self.receding_horizon(self.players[self.pl_idx])
        # if self.sempc_solver.plotter.plot:
        self.sempc_solver.plotter.plot_gp(self.players[self.pl_idx].Cx_model)
        self.one_step_planner()
        if not self.goal_in_pessi:
            print(
                "Uncertainity at meas_loc",
                self.players[self.pl_idx].get_width_at_curr_loc(),
            )
            TrainAndUpdateConstraint(
                self.players[self.pl_idx].current_location[: self.x_dim - 1],
                self.min_dist,
                self.pl_idx,
                self.players,
                self.params,
            )
            print(
                "Uncertainity at meas_loc",
                self.players[self.pl_idx].get_width_at_curr_loc(),
            )
