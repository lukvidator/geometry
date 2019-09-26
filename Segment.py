import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Line import Line
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

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self._points) + ")"

    __repr__ = __str__

    def point_by_parameter(self, t):
        return self._points[0] + t * Vector(*self._points)

    def parameter_by_point(self, point):
        vectors = [Vector(point, self._points[0]), Vector(*self._points)]
        if Vector.are_collinear(*vectors):
            return vectors[0].norm() / vectors[1].norm()
        else:
            raise ValueError("Can't get parameter of a point, which is not on the segment")

    @staticmethod
    def relation(segment1, segment2) -> (-1, 0, 1):
        """
        :param segment1: Segment object
        :param segment2: Segment object
        :return:
        1 -- if segments are intersected,
        0 -- if segments aren't parallel and not intersected,
        -1 -- otherwise
        """
        line1 = Line.from_segment(segment1)
        case = Line.relation(line1, Line.from_segment(segment2))
        if case == 0:
            return 1 if line1(segment2[0]) * line1(segment2[1]) <= 0 else 0
        else:
            return -1

    def plot(self, ax, **kwargs):
        return ax.plot([self._points[0][0], self._points[1][0]], [self._points[0][1], self._points[1][1]], **kwargs)
