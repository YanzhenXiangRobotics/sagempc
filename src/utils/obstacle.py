import math
import numpy as np


class Obstacle:
    def __init__(self, resolution) -> None:
        self.resolution = resolution

    @property
    def occup(self):
        return set()


class Rectangle(Obstacle):
    def __init__(self, lower_left, upper_right, resolution) -> None:
        super().__init__(resolution)
        assert (lower_left[0] < upper_right[0]) and (lower_left[1] < upper_right[1])
        self.lower_left = lower_left
        self.upper_right = upper_right

    @property
    def occup(self):
        occup = set()
        for x in np.arange(self.lower_left[0], self.upper_right[0], self.resolution):
            for y in np.arange(
                self.lower_left[1], self.upper_right[1], self.resolution
            ):
                occup.add((x, y))
        return occup


class Circle(Obstacle):  # assume diameter to be odd
    def __init__(self, center, radius: float, resolution) -> None:
        super().__init__(resolution)
        self.center = center
        self.radius = radius

    @property
    def occup(self):
        occup = set()
        lower_left = (
            self.center[0] - self.radius,
            self.center[1] - self.radius,
        )
        upper_right = (
            self.center[0] + self.radius,
            self.center[1] + self.radius,
        )
        X = lambda y: math.sqrt(self.radius**2 - y**2)
        for x in np.arange(lower_left[0], upper_right[0], self.resolution):
            for y in np.arange(lower_left[1], upper_right[1], self.resolution):
                if (x - self.center[0]) ** 2 + (
                    y - self.center[1]
                ) ** 2 <= self.radius**2:
                    occup.add((x, y))
        return occup
