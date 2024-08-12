from world import World
from obstacle import Rectangle, Circle
import numpy as np
import matplotlib.pyplot as plt
import os, sys
dir_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(dir_here, ".."))
from fake_simulation_node import FakeSimulationNode

import rclpy
rclpy.init()
sim = FakeSimulationNode()
world = sim.world

# fig, ax = world.plot(show=False)
min_dist_list = []
for grid in world.grids_2d():
    min_dist, _ = world.min_dist_to_obsc(grid)
    min_dist_list.append(min_dist)

min_dist_list = np.array(min_dist_list)

fig_3D = plt.figure()
ax_3D = fig_3D.add_subplot(111, projection="3d")
X1, X2 = world.grid_x1_x2()
min_dist_list = min_dist_list.reshape(len(X2), len(X1))
X1, X2 = np.meshgrid(X1, X2)
ax_3D.plot_surface(X1, X2, min_dist_list)
fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.set_xlim([world.bbox[0], world.bbox[2]])
ax.set_ylim([world.bbox[1], world.bbox[3]])
ax.contour(X1, X2, min_dist_list, levels=[0])
plt.show()

import pickle
with open("block_world_dist.pt", "wb") as f:
    pickle.dump(min_dist_list, f)