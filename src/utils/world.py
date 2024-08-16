import sys, os

dir_here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(dir_here)

from obstacle import *
import matplotlib.pyplot as plt
import numpy as np

import torch


class World:
    def __init__(self, bbox, resolution) -> None:
        self.bbox = bbox
        self.resolution = resolution

        self.obstacles = []
        self.occup = set()

    def grid_x1_x2(self, resolution=None):
        if resolution is None:
            resolution = self.resolution
        return np.arange(
            self.bbox[0], self.bbox[2], resolution
        ), np.arange(self.bbox[1], self.bbox[3], resolution)

    def grids_2d(self, resolution=None):
        if resolution is None:
            resolution = self.resolution
        X1, X2 = self.grid_x1_x2(resolution)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.vstack([X1.ravel(), X2.ravel()]).T

        return X

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)
        self.occup = self.occup.union(obstacle.occup)

    def plot(self, show=True):
        fig, ax = plt.subplots()
        occup = np.array(list(self.occup))
        ax.scatter(occup[:, 0], occup[:, 1], c="red")
        # if hasattr(self, "agent_pos"):
        #     ax.scatter(self.agent_pos[0], self.agent_pos[1], marker="x", s=30)
        #     ax.scatter(
        #         self.agent2obsc_closest_point[0],
        #         self.agent2obsc_closest_point[1],
        #         c="cyan",
        #         s=10,
        #     )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlim([self.bbox[0], self.bbox[2]])
        ax.set_ylim([self.bbox[1], self.bbox[3]])
        if show:
            plt.show()
        return fig, ax

    def min_dist_to_obsc(self, pos):
        pos = tuple(pos)
        occup_arr = np.array(list(self.occup))
        pos_arr = np.array(pos)
        dists = np.linalg.norm(occup_arr - pos_arr, axis=-1)
        min_dist = np.min(dists)
        if min_dist <= self.resolution / math.sqrt(2):
            min_dist = 0.0
        min_dist_index = np.argmin(dists)
        min_dist_point = occup_arr[min_dist_index]
        return min_dist, min_dist_point


if __name__ == "__main__":
    world = World(width=50, height=30)
    world.add_obstacle(Rectangle(lower_left=(2, 3), upper_right=(5, 7)))
    world.add_obstacle(Circle(center=(11, 14), radius=8))

    world.set_agent_pos((0, 10))
    print(world.agent2obsc_min_dist)
    world.plot()
