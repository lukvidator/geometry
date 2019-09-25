import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Point import Point
from Vector import Vector


class Segment:
    def __init__(self, points):
        self.points = points

    @classmethod
    def from_point_and_vector(cls, point, vector):
        return cls((lambda p, v: [p, p + v])(Point(point), Vector(vector)))

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if len(points) == 2:
            self._points = [Point(point) for point in points]
        else:
            raise WrongTypeException(f"Can't set {type(self)} points with {points}")

    def __getitem__(self, item):
        return self._points[item]

    def __setitem__(self, key, value):
        self._points[key] = Point(value)
