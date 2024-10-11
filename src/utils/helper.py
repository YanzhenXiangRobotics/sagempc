import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import torch
import numpy as np


def get_frame_writer():
    # FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = manimation.FFMpegWriter(
        fps=10, codec="libx264", metadata=metadata
    )  # libx264 (good quality), mpeg4
    return writer


def oracle(players, init_safe, params):
    associate_dict = {}
    associate_dict[0] = []
    for idx in range(params["env"]["n_players"]):
        associate_dict[0].append(idx)
    pessi_associate_dict = associate_dict.copy()

    # xn_star_mat, acq_val_mat = get_associated_maximizer_goal(
    #     players, associate_dict)
    Update_goal(players, associate_dict, xn_star_mat)

    return associate_dict, pessi_associate_dict, acq_val_mat


def Update_goal(players, associate_dict, xn_star_mat):
    # for key, xn_star in zip(associate_dict, xn_star_mat):
    #     for agent, xi_star in zip(associate_dict[key], xn_star):
    #         # Share the associate dict
    #         players[agent].get_expected_disc_loc(xi_star)
    #         players[agent].update_disc_boundary(xi_star)

    # covered_nodes = []  # outside of optimistc set agent can't cover
    # for key, player in enumerate(players):
    #     player.set_condi_disc_nodes(list(covered_nodes))
    #     covered_nodes = covered_nodes + player.full_disc_nodes

    list_meas_loc = []
    for key, xn_star in zip(associate_dict, xn_star_mat):
        for agent, xi_star in zip(associate_dict[key], xn_star):
            players[agent].set_others_meas_loc(
                list_meas_loc
            )  # to effi. calculate new loc
            players[agent].set_maximizer_goal(xi_star)
            list_meas_loc.append(players[agent].planned_measure_loc)


def TrainAndUpdateConstraint(query_pt, agent_key, players, params, env):
    if not torch.is_tensor(query_pt):
        query_pt = torch.from_numpy(query_pt).float()
    # 1) Fit a model on the available data based
    train = {}
    train["Cx_X"] = query_pt.reshape(-1, params["common"]["dim"])
    train["Cx_Y"] = env.get_constraint_observation(train["Cx_X"])

    players[agent_key].update_Cx_gp(train["Cx_X"], train["Cx_Y"])
    for i in range(params["env"]["n_players"]):
        if i is not agent_key:
            players[i].communicate_constraint([train["Cx_X"]], [train["Cx_Y"]])


def TrainAndUpdateConstraint_isaac_sim(
    query_pt, query_meas, agent_key, players, params
):
    # print(f"Update at {query_pt} with {query_meas}")
    # print("query_pts: ", query_pt)
    # print("query_meas: ", query_meas)
    if not torch.is_tensor(query_pt):
        query_pt = torch.from_numpy(query_pt).float()
    # 1) Fit a model on the available data based
    train = {}
    train["Cx_X"] = query_pt.reshape(-1, params["common"]["dim"])
    if not torch.is_tensor(query_meas):
        query_meas = torch.from_numpy(np.atleast_1d(query_meas)).float().reshape(-1, 1)
    train["Cx_Y"] = query_meas

    players[agent_key].update_Cx_gp_local(train["Cx_X"], train["Cx_Y"])
    for i in range(params["env"]["n_players"]):
        if i is not agent_key:
            players[i].communicate_constraint([train["Cx_X"]], [train["Cx_Y"]])
