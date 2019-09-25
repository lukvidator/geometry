import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Point import Point
from Vector import Vector


class Segment:
    def __init__(self, points):
        self.points = points

    @classmethod
    def from_point_and_vector(cls, pair):
        if len(pair) == 2:
            return cls((lambda point, vector: [point, point + vector])(Point(pair[0]), Vector(pair[1])))
        else:
            raise WrongTypeException(
                f"2 elements in arg for {cls}.from_point_and_vector expected, but {len(pair)} were given"
            )

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if len(points) == 2:
            self._points = [Point(point) for point in points]
        else:
            raise WrongTypeException(f"Can't set {type(self)} points with {points}")

    def first(self):
        return self._points[0]

    def last(self):
        return self._points[-1]

    # TODO: implement Segment class
