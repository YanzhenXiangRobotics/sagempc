from world import World
from obstacle import Rectangle, Circle
import numpy as np
import matplotlib.pyplot as plt
import os, sys

dir_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(dir_here, ".."))
from fake_simulation_node import FakeSimulationNode

import rclpy
import pickle

import gpytorch
from gpytorch.kernels import (
    MaternKernel,
)
from gpytorch.constraints import GreaterThan

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(GPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

rclpy.init()
sim = FakeSimulationNode()
world = sim.world


# fig, ax = world.plot(show=False)
store = False
if store:
    min_dist_list = []
    grids = world.grids_list()
    assert grids.shape[0] == 95 * 95
    for i, grid in enumerate(grids):
        if i % 100 == 0:
            print(i)
        min_dist, _ = world.min_dist_to_obsc(grid)
        min_dist_list.append(min_dist)

    min_dist_list = np.array(min_dist_list)

    fig_3D = plt.figure()
    ax_3D = fig_3D.add_subplot(111, projection="3d")
    X1, X2 = world.grid_x1_x2()
    min_dist_list_2D = min_dist_list.reshape(len(X2), len(X1))
    X1, X2 = np.meshgrid(X1, X2)
    ax_3D.plot_surface(X1, X2, min_dist_list_2D)
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlim([world.bbox[0], world.bbox[2]])
    ax.set_ylim([world.bbox[1], world.bbox[3]])
    ax.contour(X1, X2, min_dist_list_2D, levels=[0])
    show = True
    if show:
        plt.show()
    with open("block_world_dist.pt", "wb") as f:
        pickle.dump(min_dist_list_2D, f)
else:
    with open("block_world_dist.pt", "rb") as f:
        min_dist_list_2D = pickle.load(f)

train_gp = True
if train_gp:
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import torch
    from torch.optim import SGD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    grids_list = world.grids_list()
    grids_list -= grids_list.mean(axis=0)
    X_train = torch.tensor(grids_list, dtype=dtype).to(device)
    min_dist_list = min_dist_list_2D.flatten()
    y_train = torch.tensor(min_dist_list, dtype=dtype).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPModel(X_train, y_train.reshape(-1, 1), likelihood).to(device)
    model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    mll = mll.to(X_train).to(device)
    optimizer = SGD([{"params": model.parameters()}], lr=0.1)
    model.train()
    likelihood.train()
    NUM_EPOCHS = 500
    for epoch in range(NUM_EPOCHS):
        # clear gradients
        optimizer.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output = model(X_train)
        # Compute negative marginal log likelihood
        loss = -mll(output, y_train)
        # back prop gradients
        loss.backward()
        # print every 10 iterations
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
                f"lengthscale: {model.covar_module.lengthscale} "
                f"noise: {model.likelihood.noise.item():>4.3f}"
            )
        optimizer.step()